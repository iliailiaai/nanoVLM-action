# models/vision_language_model_action.py
from dataclasses import asdict
from typing import Optional, Tuple, Dict, Any, List

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

from safetensors.torch import load_model, save_model

from models.vision_transformer import ViT
from models.language_model import LanguageModel
from models.modality_projector import ModalityProjector
from models.config import VLMConfig
from data.processors import get_tokenizer

# Если у тебя уже есть где-то top_k_top_p_filtering — импортируй оттуда.
# Ниже простая заглушка: оставь как есть, если функция уже доступна в твоём проекте.
def top_k_top_p_filtering(logits, top_k=50, top_p=0.9):
    # минимальная реализация; если у тебя есть своя — лучше используй её.
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        values, _ = torch.topk(logits, top_k)
        min_values = values[..., -1, None]
        logits = torch.where(logits < min_values, torch.full_like(logits, -float("inf")), logits)

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumprobs = torch.cumsum(probs, dim=-1)

        # mask tokens with cum prob above top_p
        sorted_mask = cumprobs > top_p
        # keep at least 1 token
        sorted_mask[..., 0] = False

        # scatter back to original order
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask.scatter_(dim=-1, index=sorted_indices, src=sorted_mask)
        logits = torch.where(mask, torch.full_like(logits, -float("inf")), logits)

    return logits

def apply_repetition_penalty(logits, generated_ids, penalty=1.2):
    """
    logits: [B, V]
    generated_ids: [B, t]
    """
    if generated_ids is None or generated_ids.numel() == 0:
        return logits

    for b in range(logits.size(0)):
        for token_id in generated_ids[b].tolist():
            if logits[b, token_id] < 0:
                logits[b, token_id] *= penalty
            else:
                logits[b, token_id] /= penalty

    return logits



class VLReasoningActionModel(nn.Module):
    """
    nanoVLM-style VLM, but outputs action logits (3 classes) and trains with CE loss on action labels.

    Frozen (обычно):
      - vision_encoder
      - decoder
    Trainable:
      - MP
      - action_head

    New behavior:
      Optionally generates reasoning first, then pools hidden states from prompt+reasoning
      and passes pooled vector to action_head.
    """

    def __init__(self, cfg: VLMConfig, load_backbone: bool = True, num_actions: int = 3):
        super().__init__()
        self.cfg = cfg

        if load_backbone:
            print("Loading from backbone weights")
            self.vision_encoder = ViT.from_pretrained(cfg)
            self.decoder = LanguageModel.from_pretrained(cfg)
        else:
            self.vision_encoder = ViT(cfg)
            self.decoder = LanguageModel(cfg)

        self.MP = ModalityProjector(cfg)
        self.tokenizer = get_tokenizer(cfg.lm_tokenizer, cfg.vlm_extra_tokens, cfg.lm_chat_template)

        self.action_head = nn.Linear(cfg.lm_hidden_dim, num_actions, bias=True)

    def _replace_img_tokens_with_embd(self, input_ids, token_embd, image_embd):
        updated_token_embd = token_embd.clone()
        mask = (input_ids == self.tokenizer.image_token_id)  # [B, T]
        updated_token_embd[mask] = image_embd.view(-1, image_embd.size(-1)).to(updated_token_embd.dtype)
        return updated_token_embd

    def _process_images(self, images, device):
        # nanoVLM uses "images" as list of tensors (potentially nested list)
        if isinstance(images, list):
            if images and isinstance(images[0], list):
                images = [img for sublist in images for img in sublist]
            if not images:
                return None
            return torch.cat(images, dim=0).to(device)
        return images

    def _embed_prompt(self, input_ids, images, attention_mask=None):
        """
        Build token_embd for the *prompt* (text tokens with <|image|> replaced by projected vision embeddings).
        Returns: token_embd [B,T,D], attention_mask [B,T], device
        """
        device = input_ids.device
        images_tensor = self._process_images(images, device)

        token_embd = self.decoder.token_embedding(input_ids)  # [B, T, D]

        if images_tensor is not None:
            image_embd = self.vision_encoder(images_tensor)   # [N_img, T_img, D_vit]
            image_embd = self.MP(image_embd)                  # [N_img, T_img_proj, D_lm]
            token_embd = self._replace_img_tokens_with_embd(input_ids, token_embd, image_embd)

        if attention_mask is None:
            attention_mask = torch.ones(input_ids.size(0), input_ids.size(1), device=device, dtype=torch.long)

        return token_embd, attention_mask

    def _masked_mean_pool(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        hidden: [B, T, D]
        mask:   [B, T] (1 = keep, 0 = pad)
        returns: [B, D]
        """
        mask = mask.to(hidden.dtype)  # [B,T]
        denom = mask.sum(dim=1).clamp(min=1.0)  # [B]
        pooled = (hidden * mask.unsqueeze(-1)).sum(dim=1) / denom.unsqueeze(-1)
        return pooled

    @torch.inference_mode()
    def generate_reasoning_ids(
        self,
        input_ids,
        images,
        attention_mask=None,
        max_new_tokens: int = 32,
        top_k: int = 50,
        top_p: float = 0.9,
        temperature: float = 0.7,
        greedy: bool = False,
        stop_on_eos: bool = True,
        repetition_penalty=1.2
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generates reasoning tokens autoregressively and returns:
          reasoning_ids: [B, R]
          hidden_prompt: [B, T_prompt, D]
          hidden_reason: [B, R, D]

        NOTE: inference_mode -> no gradients through generation decisions.
        """
        device = input_ids.device
        token_embd, attention_mask = self._embed_prompt(input_ids, images, attention_mask=attention_mask)
        batch_size = input_ids.size(0)

        # --- Prefill ---
        prefill_output, kv_cache_list = self.decoder(
            token_embd,
            attention_mask=attention_mask,
            kv_cache=None,
            start_pos=0
        )  # prefill_output: [B, T_prompt, D]

        hidden_prompt = prefill_output
        current_total_seq_len = token_embd.size(1)

        last_token_output = prefill_output[:, -1, :]  # [B,D]
        if not self.decoder.lm_use_tokens:
            current_logits = self.decoder.head(last_token_output)  # [B,V]
        else:
            current_logits = last_token_output  # (если decoder уже выдаёт logits-эмбеддинги)

        newly_generated_ids = []
        newly_generated_hidden = []

        eos_id = self.tokenizer.eos_token_id

        for _ in range(max_new_tokens):
            if greedy:
                next_token_id = torch.argmax(current_logits, dim=-1, keepdim=True)  # [B,1]
            else:
                if newly_generated_ids:
                    generated_so_far = torch.cat(newly_generated_ids, dim=1)
                else:
                    generated_so_far = None

                current_logits = apply_repetition_penalty(
                    current_logits,
                    generated_so_far,
                    penalty=repetition_penalty
                )

                filtered = top_k_top_p_filtering(current_logits, top_k=top_k, top_p=top_p)

                probs = torch.softmax(filtered / temperature, dim=-1)

                next_token_id = torch.multinomial(probs, num_samples=1)


            newly_generated_ids.append(next_token_id)

            next_token_embed = self.decoder.token_embedding(next_token_id)  # [B,1,D]

            current_token_start_pos = current_total_seq_len
            current_total_seq_len += 1

            # extend attention mask by 1
            attention_mask = torch.cat(
                (attention_mask, torch.ones((batch_size, 1), device=device, dtype=attention_mask.dtype)),
                dim=1
            )

            decode_out, kv_cache_list = self.decoder(
                next_token_embed,
                attention_mask=attention_mask,
                kv_cache=kv_cache_list,
                start_pos=current_token_start_pos
            )  # [B,1,D]

            # save hidden for this generated token
            newly_generated_hidden.append(decode_out)  # [B,1,D]

            last_token_output = decode_out[:, -1, :]  # [B,D]
            if not self.decoder.lm_use_tokens:
                current_logits = self.decoder.head(last_token_output)
            else:
                current_logits = last_token_output

            if stop_on_eos and eos_id is not None:
                if (next_token_id.squeeze(1) == eos_id).all():
                    break

        if newly_generated_ids:
            reasoning_ids = torch.cat(newly_generated_ids, dim=1)  # [B,R]
            hidden_reason = torch.cat(newly_generated_hidden, dim=1)  # [B,R,D]
        else:
            reasoning_ids = torch.empty((batch_size, 0), dtype=torch.long, device=device)
            hidden_reason = torch.empty((batch_size, 0, self.cfg.lm_hidden_dim), dtype=hidden_prompt.dtype, device=device)

        return reasoning_ids, hidden_prompt, hidden_reason

    def forward(
        self,
        input_ids,
        images,
        attention_mask=None,
        action_labels=None,
        *,
        do_reasoning: bool = False,
        max_reasoning_tokens: int = 32,
        reasoning_top_k: int = 50,
        reasoning_top_p: float = 0.9,
        reasoning_temperature: float = 0.7,
        reasoning_greedy: bool = False,
        reasoning_repetition_penalty=1.2,
        stop_on_eos: bool = True,
        pool: str = "masked_mean",
        verbose_reasoning: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
        """
        Returns:
          action_logits: [B,3]
          loss:          scalar or None
          extra: dict with optional reasoning outputs

        If do_reasoning=False:
          - behaves like before (pool only over prompt tokens; default pool=masked_mean or last token)

        If do_reasoning=True:
          - generates reasoning first (in inference_mode, no grad through token choice),
          - pools over prompt+reasoning hidden states, then action_head.
        """
        device = input_ids.device

        extra: Dict[str, Any] = {}

        if not do_reasoning:
            # original-ish path (but using pooling option)
            token_embd, attention_mask = self._embed_prompt(input_ids, images, attention_mask=attention_mask)
            hidden, _ = self.decoder(token_embd, attention_mask=attention_mask)  # [B,T,D]

            if pool == "last":
                last_idx = attention_mask.sum(dim=1).clamp(min=1) - 1
                pooled = hidden[torch.arange(hidden.size(0), device=device), last_idx]  # [B,D]
            else:
                pooled = self._masked_mean_pool(hidden, attention_mask)

            action_logits = self.action_head(pooled)
        else:
            # reasoning-first path
            reasoning_ids, hidden_prompt, hidden_reason = self.generate_reasoning_ids(
                input_ids=input_ids,
                images=images,
                attention_mask=attention_mask,
                max_new_tokens=max_reasoning_tokens,
                top_k=reasoning_top_k,
                top_p=reasoning_top_p,
                temperature=reasoning_temperature,
                greedy=reasoning_greedy,
                stop_on_eos=stop_on_eos,
                repetition_penalty=reasoning_repetition_penalty
            )

            # Build combined hidden + mask
            B = hidden_prompt.size(0)
            T_prompt = hidden_prompt.size(1)
            R = hidden_reason.size(1)

            if attention_mask is None:
                prompt_mask = torch.ones((B, T_prompt), device=device, dtype=torch.long)
            else:
                prompt_mask = attention_mask

            if R > 0:
                reason_mask = torch.ones((B, R), device=device, dtype=torch.long)
                hidden_all = torch.cat([hidden_prompt, hidden_reason], dim=1)  # [B, T_prompt+R, D]
                mask_all = torch.cat([prompt_mask, reason_mask], dim=1)        # [B, T_prompt+R]
            else:
                hidden_all = hidden_prompt
                mask_all = prompt_mask

            if pool == "last":
                # last token over prompt+reasoning = last non-pad (reasoning is always non-pad here)
                last_idx = mask_all.sum(dim=1).clamp(min=1) - 1
                pooled = hidden_all[torch.arange(B, device=device), last_idx]
            else:
                pooled = self._masked_mean_pool(hidden_all, mask_all)

            action_logits = self.action_head(pooled)

            extra["reasoning_ids"] = reasoning_ids.detach().cpu() if verbose_reasoning else reasoning_ids
            if verbose_reasoning:
                # decode reasoning text (remove special tokens)
                # reasoning_ids is on device; decode expects CPU list
                ids_cpu = reasoning_ids.detach().cpu().tolist()
                extra["reasoning_text"] = [self.tokenizer.decode(x, skip_special_tokens=True) for x in ids_cpu]

        loss = None
        if action_labels is not None:
            loss = F.cross_entropy(action_logits, action_labels)

        return action_logits, loss, extra

    def freeze_backbones(self):
        for p in self.vision_encoder.parameters():
            p.requires_grad = False
        for p in self.decoder.parameters():
            p.requires_grad = False

    def save_pretrained(self, path: str):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "config.json"), "w", encoding="utf-8") as f:
            json.dump(asdict(self.cfg), f, ensure_ascii=False, indent=2)
        save_model(self, os.path.join(path, "model.safetensors"))

    @classmethod
    def from_pretrained(cls, repo_id_or_path: str, *, revision: Optional[str] = None) -> "VisionLanguageActionModel":
        config_path = os.path.join(repo_id_or_path, "config.json")
        weights_path = os.path.join(repo_id_or_path, "model.safetensors")
        if not (os.path.exists(config_path) and os.path.exists(weights_path)):
            raise ValueError(f"Expected {config_path} and {weights_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            cfg_dict = json.load(f)

        cfg = VLMConfig(**cfg_dict)
        model = cls(cfg, load_backbone=False)
        load_model(model, weights_path)
        return model