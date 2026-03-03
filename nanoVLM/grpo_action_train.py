#!/usr/bin/env python3
# grpo_action_train.py
#
# GRPO fine-tuning for MiniGrid EmptyEnv with an action-only policy (3 actions).
# Starts from an SFT checkpoint of VisionLanguageActionModel.
#
# Assumes you already have:
# - models/vision_language_model_action.py  (VisionLanguageActionModel)
# - data/processors.py (get_image_processor, get_image_string)
# - data/emptyenv_action_dataset.py (DEFAULT_PROMPT)
#
''' Usage example:

python grpo_action_train.py \
   --init_ckpt checkpoints_emptyenv_action/final \
   --out checkpoints_emptyenv_grpo \
   --sizes 8 9 10 \
   --rollout_episodes 32 \
   --ppo_epochs 2 \
   --minibatch_episodes 8 \
   --max_steps_per_ep 200 \
   --temperature 1.5 \
   --lr 1e-4 \
   --train_iters 200
'''

import os
import time
import math
import random
import argparse
from dataclasses import dataclass
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image

from minigrid.envs.empty import EmptyEnv

import models.config as config
from models.vision_language_model_action import VisionLanguageActionModel
from data.processors import get_image_processor, get_image_string
from data.emptyenv_action_dataset import DEFAULT_PROMPT

# silence tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

ID2ACTION = {0: "left", 1: "right", 2: "forward"}


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
    # We store the model inputs so we can recompute logits during PPO/GRPO update.
    input_ids: torch.Tensor          # [T] long
    attention_mask: torch.Tensor     # [T] long
    processed_image: list            # list[Tensor] chunks
    action: int
    logp_old: float


@dataclass
class EpisodeRecord:
    steps: list[StepRecord]
    ep_return: float
    ep_len: int
    success: int


@torch.no_grad()
def build_model_inputs_from_rgb(rgb, tokenizer, image_processor, vlm_cfg, prompt: str, device):
    """
    Convert env RGB frame to (input_ids, attention_mask, processed_image) exactly like your dataset.
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

    input_ids = torch.tensor(conv["input_ids"], dtype=torch.long, device=device)
    attention_mask = torch.tensor(conv["attention_mask"], dtype=torch.long, device=device)
    return input_ids, attention_mask, processed_image


@torch.no_grad()
def act_and_logp(model, input_ids, attention_mask, processed_image, temperature: float, greedy: bool):
    """
    Returns: action(int), logp(float), probs(np[3])
    """
    # batchify
    logits, _ = model(
        input_ids=input_ids.unsqueeze(0),
        attention_mask=attention_mask.unsqueeze(0),
        images=[[processed_image]],
        action_labels=None,
    )
    logits = logits[0]  # [3]
    if greedy:
        probs = torch.softmax(logits, dim=-1)
        action = int(torch.argmax(probs).item())
        logp = float(torch.log(probs[action] + 1e-12).item())
        return action, logp, probs.detach().cpu().numpy()

    T = float(temperature)
    if T <= 0:
        raise ValueError("temperature must be > 0")
    probs = torch.softmax(logits / T, dim=-1)
    action = int(torch.multinomial(probs, num_samples=1).item())
    logp = float(torch.log(probs[action] + 1e-12).item())
    return action, logp, probs.detach().cpu().numpy()


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
    temperature: float,
    greedy: bool,
    seed: int,
    render: bool,
):
    """
    Collect a batch of episodes using the current policy.
    """
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
                rgb, tokenizer, image_processor, vlm_cfg, prompt, device
            )
            action, logp_old, probs = act_and_logp(
                model, input_ids, attention_mask, processed_image,
                temperature=temperature, greedy=greedy
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

            if ep_len >= max_steps_per_ep:
                break

        success = 1 if ep_ret > 0 else 0
        episodes.append(EpisodeRecord(steps=steps, ep_return=ep_ret, ep_len=ep_len, success=success))
        env.close()

    return episodes


def compute_advantages(episodes: list[EpisodeRecord], baseline: str = "batch_mean"):
    """
    Advantage per step. For EmptyEnv, reward is usually sparse; simplest is:
    A = R_ep - baseline
    and assign to every step in the episode.

    Returns:
      advantages: list[list[float]] aligned with episodes[ep].steps
      stats dict
    """
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

    # normalize advantages (helps stability)
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
    """
    Flatten (episode, step) into a list for minibatching.
    """
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
):
    """
    One PPO/GRPO-style update pass over items (minibatch already chosen).
    items: list[(StepRecord, advantage)]
    """
    if not items:
        return 0.0, 0.0, 0.0

    # Build a batch. Note: images are variable nested lists; keep as python objects.
    input_ids = [s.input_ids for (s, _) in items]
    attention_mask = [s.attention_mask for (s, _) in items]
    images = [s.processed_image for (s, _) in items]
    actions = torch.tensor([s.action for (s, _) in items], dtype=torch.long, device=device)
    logp_old = torch.tensor([s.logp_old for (s, _) in items], dtype=torch.float32, device=device)
    adv = torch.tensor([a for (_, a) in items], dtype=torch.float32, device=device)

    # Pad/truncate tokens to the max length in this minibatch (simple left-pad).
    max_len = max(x.numel() for x in input_ids)
    # (Optional) cap to something reasonable if you want:
    # max_len = min(max_len, 1536)

    def left_pad(x, pad_value):
        if x.numel() < max_len:
            return F.pad(x, (max_len - x.numel(), 0), value=pad_value)
        elif x.numel() > max_len:
            return x[-max_len:]
        return x

    input_ids = torch.stack([left_pad(x, model.tokenizer.pad_token_id) for x in input_ids]).long()
    attention_mask = torch.stack([left_pad(x, 0) for x in attention_mask]).long()

    # Forward
    model.train()
    logits, _ = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        images=[[im] for im in images],   # batch structure: list of packs
        action_labels=None,
    )  # [B,3]

    logp = torch.log_softmax(logits, dim=-1)  # [B,3]
    logp_a = logp.gather(1, actions.view(-1, 1)).squeeze(1)  # [B]

    ratio = torch.exp(logp_a - logp_old)  # [B]
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    policy_loss = -torch.mean(torch.minimum(unclipped, clipped))

    # Entropy bonus (encourage exploration)
    probs = torch.softmax(logits, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1).mean()
    loss = policy_loss - entropy_coef * entropy

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if grad_clip is not None and grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

    # simple diagnostics
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
    parser.add_argument("--out", type=str, default="checkpoints_emptyenv_grpo", help="Output dir for checkpoints")

    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--sizes", type=int, nargs="+", default=[8], help="Env sizes to train on, e.g. --sizes 6 7 8")
    parser.add_argument("--max_steps_per_ep", type=int, default=200)

    parser.add_argument("--train_iters", type=int, default=200, help="How many rollout+update iterations")
    parser.add_argument("--rollout_episodes", type=int, default=32, help="Episodes per iteration")
    parser.add_argument("--ppo_epochs", type=int, default=2, help="Update epochs per iteration")
    parser.add_argument("--minibatch_episodes", type=int, default=8, help="Episodes per minibatch")
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--entropy_coef", type=float, default=0.01)

    parser.add_argument("--temperature", type=float, default=1.5, help="Sampling temperature during rollouts")
    parser.add_argument("--greedy", action="store_true", help="Use greedy actions during rollouts (NOT recommended for training)")

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=1, help="Log every N train iters")
    parser.add_argument("--eval_every", type=int, default=10, help="Eval (greedy) every N train iters")
    parser.add_argument("--eval_episodes", type=int, default=50)
    parser.add_argument("--render_rollouts", action="store_true")

    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print("Device:", device)

    os.makedirs(args.out, exist_ok=True)

    # Config + model
    vlm_cfg = config.VLMConfig()
    vlm_cfg.max_img_size = 512

    model = VisionLanguageActionModel.from_pretrained(args.init_ckpt)
    model.to(device)
    model.train()

    # Usually you want to keep backbones frozen during RL too (for stability/speed)
    if hasattr(model, "freeze_backbones"):
        model.freeze_backbones()

    tokenizer = model.tokenizer
    image_processor = get_image_processor(vlm_cfg.max_img_size, vlm_cfg.vit_img_size, vlm_cfg.resize_to_max_side_len)

    # Optimizer: only train MP + head (common for this setup)
    train_params = []
    if hasattr(model, "MP"):
        train_params += list(model.MP.parameters())
    if hasattr(model, "action_head"):
        train_params += list(model.action_head.parameters())

    optimizer = optim.AdamW(train_params, lr=args.lr, weight_decay=args.weight_decay)

    # Initial eval
    print("Initial eval...")
    with torch.no_grad():
        eps = rollout_episodes(
            model, vlm_cfg, tokenizer, image_processor, device,
            sizes=args.sizes, prompt=args.prompt,
            rollout_episodes=args.eval_episodes,
            max_steps_per_ep=args.max_steps_per_ep,
            temperature=args.temperature, greedy=False,
            seed=args.seed + 10_000,
            render=False,
        )
    sr0 = np.mean([e.success for e in eps])
    ret0 = np.mean([e.ep_return for e in eps])
    print(f"[EVAL0] success_rate={sr0:.3f} avg_return={ret0:.3f} avg_len={np.mean([e.ep_len for e in eps]):.1f}")

    t0 = time.time()
    rng = np.random.default_rng(args.seed)

    for it in range(1, args.train_iters + 1):
        # 1) Rollouts
        episodes = rollout_episodes(
            model, vlm_cfg, tokenizer, image_processor, device,
            sizes=args.sizes, prompt=args.prompt,
            rollout_episodes=args.rollout_episodes,
            max_steps_per_ep=args.max_steps_per_ep,
            temperature=args.temperature, greedy=args.greedy,
            seed=args.seed + it * 1000,
            render=args.render_rollouts,
        )
        adv_eps, stats = compute_advantages(episodes, baseline="batch_mean")
        items = flatten_steps(episodes, adv_eps)

        # 2) Updates (GRPO/PPO-style)
        # minibatch by episodes: approximate by batching steps from a subset of episodes
        # We'll do a simple step-based minibatch size derived from minibatch_episodes.
        steps_per_ep = max(1, int(np.mean([ep.ep_len for ep in episodes])))
        minibatch_size = max(16, args.minibatch_episodes * steps_per_ep)

        losses = []
        kls = []
        clip_fracs = []

        for _ in range(args.ppo_epochs):
            for mb in minibatches(items, minibatch_size=minibatch_size, rng=rng):
                loss, kl, clip_frac = grpo_update(
                    model, optimizer, mb, device,
                    clip_eps=args.clip_eps,
                    entropy_coef=args.entropy_coef,
                    grad_clip=args.grad_clip,
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

        # 4) Periodic greedy eval + checkpoint
        if it % args.eval_every == 0:
            with torch.no_grad():
                eps = rollout_episodes(
                    model, vlm_cfg, tokenizer, image_processor, device,
                    sizes=args.sizes, prompt=args.prompt,
                    rollout_episodes=args.eval_episodes,
                    max_steps_per_ep=args.max_steps_per_ep,
                    temperature=args.temperature, greedy=False,
                    seed=args.seed + 50_000 + it,
                    render=False,
                )
            sr = np.mean([e.success for e in eps])
            ret = np.mean([e.ep_return for e in eps])
            print(f"[EVAL] it {it:4d} | success_rate={sr:.3f} avg_return={ret:.3f} avg_len={np.mean([e.ep_len for e in eps]):.1f}")

            ckpt_path = save_ckpt(model, args.out, f"it_{it}")
            print("Saved checkpoint:", ckpt_path)

    final_path = save_ckpt(model, args.out, "final")
    print("Done. Final checkpoint:", final_path)


if __name__ == "__main__":
    main()