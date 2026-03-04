# grpo_train_lora.py
# GRPO training for MiniGrid EmptyEnv with:
# - VLReasoningActionModel (reasoning -> pooling -> classification head)
# - LoRA on LLM q_proj/v_proj
# - LoRA on Modality Projector (MP) Linear layers
# - Full finetune of classification head
#
# Keeps rollout/update logic the same as your current GRPO script.

import os
import time
import math
import random
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional

from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image

from minigrid.envs.empty import EmptyEnv

import models.config as config
from models.vl_reasoning_action_model import VLReasoningActionModel
from data.processors import get_image_processor, get_image_string
from data.emptyenv_action_dataset import REASONING_PROMPT  # as requested

# silence tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

ID2ACTION = {0: "left", 1: "right", 2: "forward"}


# -------------------------
# LoRA utilities
# -------------------------
class LoRALinear(nn.Module):
    """
    Wraps a base nn.Linear with LoRA adapters.
    y = base(x) + scale * ( (x @ A^T) @ B^T )
    """
    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("LoRALinear expects nn.Linear base module")

        self.base = base
        self.in_features = base.in_features
        self.out_features = base.out_features

        self.r = int(r)
        self.alpha = int(alpha)
        self.scale = (self.alpha / self.r) if self.r > 0 else 1.0
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

        # Freeze base
        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

        if self.r > 0:
            # A: (r, in), B: (out, r)
            self.lora_A = nn.Parameter(torch.zeros(self.r, self.in_features))
            self.lora_B = nn.Parameter(torch.zeros(self.out_features, self.r))
            # init
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        if self.r <= 0:
            return y
        # (B,*,in) @ (in,r) -> (B,*,r) then @ (r,out) -> (B,*,out)
        lora = (self.dropout(x) @ self.lora_A.t()) @ self.lora_B.t()
        return y + self.scale * lora


def _get_parent_module(root: nn.Module, module_name: str) -> Tuple[nn.Module, str]:
    parts = module_name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def inject_lora_by_name(
    root: nn.Module,
    name_predicate,
    r: int,
    alpha: int,
    dropout: float,
) -> int:
    """
    Replace selected nn.Linear modules in `root` with LoRALinear.
    name_predicate(name, module) -> bool
    Returns how many modules replaced.
    """
    to_replace = []
    for name, module in root.named_modules():
        if isinstance(module, nn.Linear) and name_predicate(name, module):
            to_replace.append(name)

    replaced = 0
    for name in to_replace:
        parent, attr = _get_parent_module(root, name)
        base = getattr(parent, attr)
        setattr(parent, attr, LoRALinear(base, r=r, alpha=alpha, dropout=dropout))
        replaced += 1

    return replaced


def mark_trainable_lora_and_head(model: nn.Module):
    # freeze everything
    for p in model.parameters():
        p.requires_grad = False

    # unfreeze LoRA params
    for m in model.modules():
        if isinstance(m, LoRALinear):
            if m.lora_A is not None:
                m.lora_A.requires_grad = True
            if m.lora_B is not None:
                m.lora_B.requires_grad = True

    # unfreeze classification head fully
    if hasattr(model, "action_head"):
        for p in model.action_head.parameters():
            p.requires_grad = True


def count_trainable_params(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


# -------------------------
# RL helpers
# -------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else torch.device("cpu")
    )


@dataclass
class StepRecord:
    # store inputs so we can recompute logits during PPO update
    input_ids: torch.Tensor          # [T] long (CPU)
    attention_mask: torch.Tensor     # [T] long (CPU)
    processed_image: list            # list[Tensor] chunks (CPU tensors)
    action: int
    logp_old: float


@dataclass
class EpisodeRecord:
    steps: list
    ep_return: float
    ep_len: int
    success: int


@torch.no_grad()
def build_model_inputs_from_rgb(rgb, tokenizer, image_processor, vlm_cfg, prompt: str):
    """
    Build (input_ids, attention_mask, processed_image) exactly like dataset.
    NOTE: keeps tensors on CPU to reduce GPU memory pressure during rollouts.
    """
    img = Image.fromarray(rgb).convert("RGB")
    processed_image, splitted_image_count = image_processor(img)

    messages = [{"role": "user", "content": prompt}]
    image_string = get_image_string(tokenizer, [splitted_image_count], vlm_cfg.mp_image_token_length)
    messages[0]["content"] = image_string + messages[0]["content"]

    conv = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_special_tokens=False,
        return_dict=True,
    )

    input_ids = torch.tensor(conv["input_ids"], dtype=torch.long, device="cpu")
    attention_mask = torch.tensor(conv["attention_mask"], dtype=torch.long, device="cpu")
    return input_ids, attention_mask, processed_image


def _call_model_action_logits(model, **kwargs):
    """
    Support both:
      (logits, loss)
    and
      (logits, loss, extra)
    """
    out = model(**kwargs)
    if isinstance(out, (tuple, list)):
        if len(out) == 2:
            return out[0], None, {}
        if len(out) == 3:
            return out[0], out[1], out[2]
    # fallback
    return out, None, {}


@torch.no_grad()
def act_and_logp_reasoning(
    model,
    input_ids_cpu,
    attention_mask_cpu,
    processed_image,
    device,
    action_temperature: float,
    greedy: bool,
    # reasoning params
    reasoning_top_p: float,
    reasoning_top_k: int,
    max_reasoning_tokens: int,
    reasoning_temperature: float,
    reasoning_repetition_penalty: float,
    verbose_reasoning: bool = False,
):
    """
    Returns: action(int), logp(float), probs(np[3]), reasoning_text(optional)
    """
    input_ids = input_ids_cpu.unsqueeze(0).to(device, non_blocking=True)
    attention_mask = attention_mask_cpu.unsqueeze(0).to(device, non_blocking=True)
    images = [[processed_image]]  # batch size 1

    logits, _, extra = _call_model_action_logits(
        model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        images=images,
        action_labels=None,
        do_reasoning=True,
        max_reasoning_tokens=max_reasoning_tokens,
        reasoning_top_p=reasoning_top_p,
        reasoning_top_k=reasoning_top_k,
        reasoning_temperature=reasoning_temperature,
        reasoning_repetition_penalty=reasoning_repetition_penalty,
        reasoning_greedy=False,
        stop_on_eos=True,
        verbose_reasoning=verbose_reasoning,
    )
    logits = logits[0]  # [3]

    if greedy:
        probs_t = torch.softmax(logits, dim=-1)
        action = int(torch.argmax(probs_t).item())
        logp = float(torch.log(probs_t[action] + 1e-12).item())
    else:
        T = float(action_temperature)
        probs_t = torch.softmax(logits / max(T, 1e-6), dim=-1)
        action = int(torch.multinomial(probs_t, num_samples=1).item())
        logp = float(torch.log(probs_t[action] + 1e-12).item())

    probs = probs_t.detach().cpu().numpy()
    reasoning_text = None
    if verbose_reasoning and isinstance(extra, dict) and "reasoning_text" in extra:
        reasoning_text = extra["reasoning_text"][0]

    return action, logp, probs, reasoning_text


def rollout_episodes(
    model,
    vlm_cfg,
    tokenizer,
    image_processor,
    device,
    sizes: list[int],
    prompt: str,
    rollout_episodes: int,
    max_steps_per_ep: int,
    action_temperature: float,
    greedy: bool,
    seed: int,
    render: bool,
    # reasoning params
    reasoning_top_p: float,
    reasoning_top_k: int,
    max_reasoning_tokens: int,
    reasoning_temperature: float,
    reasoning_repetition_penalty: float,
    verbose_reasoning: bool,
):
    episodes: list[EpisodeRecord] = []
    rng = np.random.default_rng(seed)

    model.eval()

    for ep in tqdm(range(rollout_episodes)):
        size = int(rng.choice(sizes))
        env = EmptyEnv(size=size, agent_start_pos=None, render_mode="rgb_array")
        obs, info = env.reset(seed=seed + ep)
        env.agent_dir = int(rng.integers(0, 4))

        steps: list[StepRecord] = []
        ep_ret = 0.0
        ep_len = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            rgb = env.render()

            input_ids, attention_mask, processed_image = build_model_inputs_from_rgb(
                rgb, tokenizer, image_processor, vlm_cfg, prompt
            )

            action, logp_old, probs, rtext = act_and_logp_reasoning(
                model,
                input_ids,
                attention_mask,
                processed_image,
                device=device,
                action_temperature=action_temperature,
                greedy=greedy,
                reasoning_top_p=reasoning_top_p,
                reasoning_top_k=reasoning_top_k,
                max_reasoning_tokens=max_reasoning_tokens,
                reasoning_temperature=reasoning_temperature,
                reasoning_repetition_penalty=reasoning_repetition_penalty,
                verbose_reasoning=verbose_reasoning,
            )

            obs, reward, terminated, truncated, info = env.step(action)
            ep_ret += float(reward)
            ep_len += 1

            steps.append(
                StepRecord(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    processed_image=processed_image,
                    action=action,
                    logp_old=logp_old,
                )
            )

            if render:
                print(f"ep={ep} size={size} t={ep_len} a={ID2ACTION[action]} probs={np.round(probs,3)}")
                if verbose_reasoning and rtext is not None:
                    print("  reasoning:", rtext)

            if ep_len >= max_steps_per_ep:
                break

        success = 1 if ep_ret > 0 else 0
        episodes.append(EpisodeRecord(steps=steps, ep_return=ep_ret, ep_len=ep_len, success=success))
        env.close()

    return episodes


def compute_advantages(episodes: list[EpisodeRecord], baseline: str = "batch_mean"):
    returns = np.array([ep.ep_return for ep in episodes], dtype=np.float32)
    if baseline == "batch_mean":
        b = float(returns.mean())
    elif baseline == "zero":
        b = 0.0
    else:
        raise ValueError("baseline must be 'batch_mean' or 'zero'")

    adv_eps = []
    for ep in episodes:
        a = float(ep.ep_return - b)
        adv_eps.append([a] * len(ep.steps))

    flat_adv = np.array([a for adv in adv_eps for a in adv], dtype=np.float32)
    if flat_adv.size > 1:
        mu = float(flat_adv.mean())
        sd = float(flat_adv.std() + 1e-8)
        adv_eps = [[(a - mu) / sd for a in adv] for adv in adv_eps]

    stats = {
        "return_mean": float(returns.mean()),
        "return_std": float(returns.std()),
        "baseline": float(b),
        "success_rate": float(np.mean([ep.success for ep in episodes])),
        "avg_len": float(np.mean([ep.ep_len for ep in episodes])),
    }
    return adv_eps, stats


def flatten_steps(episodes: list[EpisodeRecord], adv_eps: list[list[float]]):
    items = []
    for ep_idx, ep in enumerate(episodes):
        for t, step in enumerate(ep.steps):
            items.append((step, float(adv_eps[ep_idx][t])))
    return items


def minibatches(items, minibatch_size, rng: np.random.Generator):
    idx = np.arange(len(items))
    rng.shuffle(idx)
    for start in range(0, len(items), minibatch_size):
        j = idx[start:start + minibatch_size]
        yield [items[k] for k in j]


def grpo_update(
    model,
    optimizer,
    items,
    device,
    clip_eps: float,
    entropy_coef: float,
    grad_clip: float | None,
    use_amp: bool,
    scaler: Optional[torch.cuda.amp.GradScaler],
    amp_dtype,
    max_len_cap: int,
    # reasoning params
    reasoning_top_p: float,
    reasoning_top_k: int,
    max_reasoning_tokens: int,
    reasoning_temperature: float,
    reasoning_repetition_penalty: float,
):
    if not items:
        return 0.0, 0.0, 0.0

    input_ids = [s.input_ids for (s, _) in items]           # CPU [T]
    attention_mask = [s.attention_mask for (s, _) in items] # CPU [T]
    images = [s.processed_image for (s, _) in items]        # CPU list
    actions = torch.tensor([s.action for (s, _) in items], dtype=torch.long, device=device)
    logp_old = torch.tensor([s.logp_old for (s, _) in items], dtype=torch.float32, device=device)
    adv = torch.tensor([a for (_, a) in items], dtype=torch.float32, device=device)

    max_len = max(x.numel() for x in input_ids)
    if max_len_cap > 0:
        max_len = min(max_len, max_len_cap)

    # NOTE: in your dataset/image-string, image tokens are at the HEAD.
    # So keep HEAD when truncating; right-pad.
    def right_pad_keep_head(x, pad_value):
        if x.numel() < max_len:
            return F.pad(x, (0, max_len - x.numel()), value=pad_value)
        elif x.numel() > max_len:
            return x[:max_len]
        return x

    input_ids = torch.stack([right_pad_keep_head(x, model.tokenizer.pad_token_id) for x in input_ids]).long()
    attention_mask = torch.stack([right_pad_keep_head(x, 0) for x in attention_mask]).long()

    input_ids = input_ids.to(device, non_blocking=True)
    attention_mask = attention_mask.to(device, non_blocking=True)

    model.train()

    autocast_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=(use_amp and device.type == "cuda"))
    with autocast_ctx:
        logits, _, _ = _call_model_action_logits(
            model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            images=[[im] for im in images],
            action_labels=None,
            do_reasoning=True,
            max_reasoning_tokens=max_reasoning_tokens,
            reasoning_top_p=reasoning_top_p,
            reasoning_top_k=reasoning_top_k,
            reasoning_temperature=reasoning_temperature,
            reasoning_repetition_penalty=reasoning_repetition_penalty,
            reasoning_greedy=False,
            stop_on_eos=True,
            verbose_reasoning=False,
        )

        logp = torch.log_softmax(logits, dim=-1)
        logp_a = logp.gather(1, actions.view(-1, 1)).squeeze(1)

        ratio = torch.exp(logp_a - logp_old)
        unclipped = ratio * adv
        clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
        policy_loss = -torch.mean(torch.minimum(unclipped, clipped))

        probs = torch.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1).mean()
        loss = policy_loss - entropy_coef * entropy

    optimizer.zero_grad(set_to_none=True)

    if use_amp and device.type == "cuda":
        assert scaler is not None
        scaler.scale(loss).backward()
        if grad_clip is not None and grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], grad_clip)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], grad_clip)
        optimizer.step()

    approx_kl = torch.mean(logp_old - logp_a).item()
    clip_frac = torch.mean((torch.abs(ratio - 1.0) > clip_eps).float()).item()
    return float(loss.item()), float(approx_kl), float(clip_frac)


def save_ckpt(model, out_dir: str, name: str):
    path = os.path.join(out_dir, name)
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_ckpt", type=str, required=True, help="Path to SFT checkpoint dir")
    parser.add_argument("--out", type=str, default="checkpoints_emptyenv_grpo_lora", help="Output dir for checkpoints")

    parser.add_argument("--prompt", type=str, default=REASONING_PROMPT)
    parser.add_argument("--sizes", type=int, nargs="+", default=[8], help="Env sizes to train on, e.g. --sizes 6 7 8")
    parser.add_argument("--max_steps_per_ep", type=int, default=200)

    parser.add_argument("--train_iters", type=int, default=200)
    parser.add_argument("--rollout_episodes", type=int, default=32)
    parser.add_argument("--ppo_epochs", type=int, default=2)
    parser.add_argument("--minibatch_episodes", type=int, default=8)

    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--entropy_coef", type=float, default=0.01)

    parser.add_argument("--temperature", type=float, default=1.4, help="Action sampling temperature during rollouts")
    parser.add_argument("--greedy", action="store_true")

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=1)

    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--eval_sizes", type=int, nargs="+", default=None)
    parser.add_argument("--eval_episodes", type=int, default=50)
    parser.add_argument("--render_rollouts", action="store_true")
    parser.add_argument("--verbose_reasoning", action="store_true", help="Print reasoning during rollouts (debug)")

    # memory control
    parser.add_argument("--max_len_cap", type=int, default=512, help="Cap token length per minibatch (keep HEAD, right-pad)")
    parser.add_argument("--max_img_size", type=int, default=512)

    # LoRA params
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.0)

    # Reasoning generation params (your suggested defaults)
    parser.add_argument("--reasoning_top_p", type=float, default=0.75)
    parser.add_argument("--reasoning_top_k", type=int, default=20)
    parser.add_argument("--max_reasoning_tokens", type=int, default=32)
    parser.add_argument("--reasoning_temperature", type=float, default=0.5)
    parser.add_argument("--reasoning_repetition_penalty", type=float, default=1.2)

    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print("Device:", device)
    os.makedirs(args.out, exist_ok=True)

    # config
    vlm_cfg = config.VLMConfig()
    vlm_cfg.max_img_size = args.max_img_size

    # model
    model = VLReasoningActionModel.from_pretrained(args.init_ckpt)
    model.to(device)
    model.train()

    # Freeze backbones first (vision + decoder) as you already do
    if hasattr(model, "freeze_backbones"):
        model.freeze_backbones()

    # Inject LoRA into LLM q/v
    def is_qv(name, module):
        # robust: endswith match catches nested names like "...attn.q_proj"
        return name.endswith("q_proj") or name.endswith("v_proj")

    replaced_llm = inject_lora_by_name(
        model.decoder,
        name_predicate=is_qv,
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
    )

    # Inject LoRA into MP (all Linear layers inside MP)
    replaced_mp = inject_lora_by_name(
        model.MP,
        name_predicate=lambda n, m: True,
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
    )

    # Now: train only LoRA + action_head
    mark_trainable_lora_and_head(model)

    model.to(device)

    tr, tot = count_trainable_params(model)
    print(f"LoRA injected: LLM={replaced_llm} linear(s), MP={replaced_mp} linear(s)")
    print(f"Trainable params: {tr:,} / {tot:,} ({100.0*tr/tot:.4f}%)")

    # tokenizer + image processor
    tokenizer = model.tokenizer
    image_processor = get_image_processor(vlm_cfg.max_img_size, vlm_cfg.vit_img_size, vlm_cfg.resize_to_max_side_len)

    # optimizer over trainable params only
    train_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(train_params, lr=args.lr, weight_decay=args.weight_decay)

    # AMP for fp16 (cuda)
    use_amp = (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    amp_dtype = torch.float16

    t0 = time.time()
    rng = np.random.default_rng(args.seed)

    for it in range(1, args.train_iters + 1):
        # 1) Rollouts
        episodes = rollout_episodes(
            model, vlm_cfg, tokenizer, image_processor, device,
            sizes=args.sizes,
            prompt=args.prompt,
            rollout_episodes=args.rollout_episodes,
            max_steps_per_ep=args.max_steps_per_ep,
            action_temperature=args.temperature,
            greedy=args.greedy,
            seed=args.seed + it * 1000,
            render=args.render_rollouts,
            reasoning_top_p=args.reasoning_top_p,
            reasoning_top_k=args.reasoning_top_k,
            max_reasoning_tokens=args.max_reasoning_tokens,
            reasoning_temperature=args.reasoning_temperature,
            reasoning_repetition_penalty=args.reasoning_repetition_penalty,
            verbose_reasoning=args.verbose_reasoning,
        )
        adv_eps, stats = compute_advantages(episodes, baseline="batch_mean")
        items = flatten_steps(episodes, adv_eps)

        # 2) Updates
        steps_per_ep = max(1, int(np.mean([ep.ep_len for ep in episodes])))
        minibatch_size = max(16, args.minibatch_episodes * steps_per_ep)

        losses, kls, clip_fracs = [], [], []

        for _ in range(args.ppo_epochs):
            for mb in minibatches(items, minibatch_size=minibatch_size, rng=rng):
                loss, kl, clip_frac = grpo_update(
                    model, optimizer, mb, device,
                    clip_eps=args.clip_eps,
                    entropy_coef=args.entropy_coef,
                    grad_clip=args.grad_clip,
                    use_amp=use_amp,
                    scaler=scaler,
                    amp_dtype=amp_dtype,
                    max_len_cap=args.max_len_cap,
                    reasoning_top_p=args.reasoning_top_p,
                    reasoning_top_k=args.reasoning_top_k,
                    max_reasoning_tokens=args.max_reasoning_tokens,
                    reasoning_temperature=args.reasoning_temperature,
                    reasoning_repetition_penalty=args.reasoning_repetition_penalty,
                )
                losses.append(loss)
                kls.append(kl)
                clip_fracs.append(clip_frac)

        # 3) Logging
        if it % args.log_every == 0:
            dt = time.time() - t0
            print(
                f"it {it:4d}/{args.train_iters} | "
                f"rollout_sr {stats['success_rate']:.3f} | "
                f"ret {stats['return_mean']:.3f} | "
                f"len {stats['avg_len']:.1f} | "
                f"loss {np.mean(losses):.4f} | "
                f"kl {np.mean(kls):.4f} | "
                f"clip {np.mean(clip_fracs):.3f} | "
                f"{dt:.1f}s"
            )

        # 4) Periodic checkpoint (kept as in your script)
        if it % args.eval_every == 0:
            ckpt_path = save_ckpt(model, args.out, f"it_{it}")
            print("Saved checkpoint:", ckpt_path)

    final_path = save_ckpt(model, args.out, "final")
    print("Done. Final checkpoint:", final_path)


if __name__ == "__main__":
    main()