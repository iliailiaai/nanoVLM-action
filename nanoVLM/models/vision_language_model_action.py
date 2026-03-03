# models/vision_language_model_action.py
from dataclasses import asdict
from typing import Optional

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


class VisionLanguageActionModel(nn.Module):
    """
    nanoVLM-style VLM, but outputs action logits (3 classes) and trains with CE loss on action labels.

    Frozen:
      - vision_encoder
      - decoder
    Trainable:
      - MP
      - action_head
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

    def forward(self, input_ids, images, attention_mask=None, action_labels=None):
        """
        action_labels: LongTensor [B] with values {0,1,2} mapping to left/right/forward.
        """
        device = input_ids.device
        images_tensor = self._process_images(images, device)

        # token embeddings for text
        token_embd = self.decoder.token_embedding(input_ids)  # [B, T, D]

        # insert image embeddings into token stream (at <|image|> placeholder positions)
        if images_tensor is not None:
            image_embd = self.vision_encoder(images_tensor)   # [N_img, T_img, D_vit]
            image_embd = self.MP(image_embd)                  # [N_img, T_img_proj, D_lm]
            token_embd = self._replace_img_tokens_with_embd(input_ids, token_embd, image_embd)

        # decoder output (hidden states)
        hidden, _ = self.decoder(token_embd, attention_mask=attention_mask)  # [B, T, D]

        # pick the last non-pad token per sample using attention_mask
        if attention_mask is None:
            last_idx = torch.full((hidden.size(0),), hidden.size(1) - 1, device=device, dtype=torch.long)
        else:
            # attention_mask is 1 for real tokens, 0 for pads
            last_idx = attention_mask.sum(dim=1).clamp(min=1) - 1  # [B]

        pooled = hidden[torch.arange(hidden.size(0), device=device), last_idx]  # [B, D]
        action_logits = self.action_head(pooled)  # [B, 3]

        loss = None
        if action_labels is not None:
            loss = F.cross_entropy(action_logits, action_labels)

        return action_logits, loss

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
        # local folder only (simple)
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