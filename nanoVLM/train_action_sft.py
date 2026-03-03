# train_action_sft.py
import os
import math
import time
import argparse
from dataclasses import asdict
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

import models.config as config
from models.vision_language_model_action import VisionLanguageActionModel
from data.emptyenv_action_dataset import EmptyEnvActionDataset, DEFAULT_PROMPT
from data.action_collator import ActionCollator


def set_seed(seed: int = 0):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_n = 0
    correct = 0
    print('eval...')

    for batch in loader:

        if batch["input_ids"].numel() == 0:
            continue
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        images = batch["images"]
        labels = batch["action_label"].to(device)

        logits, loss = model(input_ids=input_ids, images=images, attention_mask=attention_mask, action_labels=labels)
        total_loss += float(loss.item()) * labels.size(0)
        total_n += labels.size(0)

        pred = torch.argmax(logits, dim=-1)
        correct += int((pred == labels).sum().item())

    avg_loss = total_loss / max(1, total_n)
    acc = correct / max(1, total_n)
    return avg_loss, acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=str, required=True, help="Path to dataset.jsonl")
    parser.add_argument("--images_root", type=str, required=True, help="Root folder that contains the images/ subdir")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT, help="English task prompt")
    parser.add_argument("--out", type=str, default="checkpoints_emptyenv_action", help="Where to save checkpoints")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--lr_mp", type=float, default=5e-3)
    parser.add_argument("--lr_head", type=float, default=5e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--val_ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--eval_every", type=int, default=250)
    args = parser.parse_args()

    set_seed(args.seed)

    vlm_cfg = config.VLMConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")

    # Build dataset
    vlm_cfg.max_img_size = 512 
    full_ds = EmptyEnvActionDataset(
        jsonl_path=args.jsonl,
        images_root=args.images_root,
        vlm_cfg=vlm_cfg,
        prompt=args.prompt,
        max_samples = 20_000
    )

    test_path_json = args.jsonl.replace("emptyenv_sft_dataset/dataset.jsonl", "emptyenv_sft_dataset/test_unseen_sizes/dataset.jsonl")
    test_images_root = args.images_root.replace("emptyenv_sft_dataset", "emptyenv_sft_dataset/test_unseen_sizes/")
    print(test_path_json, test_images_root)

    test_unseen_ds = EmptyEnvActionDataset(
        jsonl_path=test_path_json,
        images_root=test_images_root,
        vlm_cfg=vlm_cfg,
        prompt=args.prompt,
        max_samples = 20_000
    )

    val_size = max(1, int(len(full_ds) * args.val_ratio))
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed))

    # Model
    model = VisionLanguageActionModel(vlm_cfg, load_backbone=True, num_actions=3)
    model.freeze_backbones()  # freeze vision + language
    model.to(device)

    print('compiling model...')
    model = torch.compile(model)

    # Collator
    collator = ActionCollator(model.tokenizer, max_length=args.max_length)

    import time

    print("Dataset length:", len(full_ds))

    t = time.time()
    print("Fetching sample 0 ...", flush=True)
    s0 = full_ds[0]
    print("Fetched sample 0 in", time.time() - t, "sec", flush=True)

    print("input_ids:", s0["input_ids"].shape, "label:", int(s0["action_label"]), flush=True)
    print("num image chunks:", len(s0["images"][0]), flush=True)

    g = torch.Generator()
    g.manual_seed(0)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,    # =per device BS in DDP
        collate_fn=collator,
        num_workers=3,
        pin_memory=True,
        persistent_workers=False,
        drop_last=True,
        generator=g,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        collate_fn=collator,
        num_workers=1,
        pin_memory=True,
        persistent_workers=False,
        drop_last=True,
        generator=g,
    )

    test_unseen_loader = DataLoader(
        test_unseen_ds,
        batch_size=args.batch_size,
        collate_fn=collator,
        num_workers=1,
        pin_memory=True,
        persistent_workers=False,
        drop_last=True,
        generator=g,
    )



    # Optimizer: only MP + head
    params = [
        {"params": list(model.MP.parameters()), "lr": args.lr_mp},
        {"params": list(model.action_head.parameters()), "lr": args.lr_head},
    ]
    optimizer = optim.AdamW(params, weight_decay=args.weight_decay)

    os.makedirs(args.out, exist_ok=True)

    # Train loop
    model.train()
    step = 0
    t0 = time.time()

    # first 
    val_loss, val_acc = evaluate(model, val_loader, device)
    unseen_sizes_loss, unseen_sizes_acc = evaluate(model, test_unseen_loader, device)
    print(f"val_loss {val_loss:.4f} | val_acc {val_acc:.3f} ")
    print(f"unseen_sizes_loss {unseen_sizes_loss:.4f} | unseen_sizes_acc {unseen_sizes_acc:.3f} ")

    while step < args.max_steps:
        for batch in tqdm(train_loader):
            if step >= args.max_steps:
                break
            if batch["input_ids"].numel() == 0:
                continue

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            images = batch["images"]
            labels = batch["action_label"].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits, loss = model(input_ids=input_ids, images=images, attention_mask=attention_mask, action_labels=labels)
            loss.backward()

            if args.grad_clip is not None and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()
            step += 1

            if step % args.log_every == 0:
                dt = time.time() - t0
                print(f"step {step:6d} | train_loss {loss.item():.4f} | {dt:.1f}s")

            if step % args.eval_every == 0:
                val_loss, val_acc = evaluate(model, val_loader, device)
                print(f"[VAL] step {step:6d} | val_loss {val_loss:.4f} | val_acc {val_acc:.3f}")
                
                unseen_sizes_loss, unseen_sizes_acc = evaluate(model, test_unseen_loader, device)
                print(f"unseen_sizes_loss {unseen_sizes_loss:.4f} | unseen_sizes_acc {unseen_sizes_acc:.3f} ")

                ckpt_dir = os.path.join(args.out, f"step_{step}")
                model.save_pretrained(ckpt_dir)
                print(f"Saved checkpoint to: {ckpt_dir}")

    # final save
    final_dir = os.path.join(args.out, "final")
    model.save_pretrained(final_dir)
    print(f"Done. Final checkpoint: {final_dir}")


if __name__ == "__main__":
    main()