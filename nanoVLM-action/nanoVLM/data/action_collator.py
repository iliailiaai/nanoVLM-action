# data/action_collator.py
import torch

class ActionCollator:

    def __init__(self, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        batch = [b for b in batch if b is not None]
        if not batch:
            return {
                "input_ids": torch.empty(0, dtype=torch.long),
                "attention_mask": torch.empty(0, dtype=torch.long),
                "images": [],
                "action_label": torch.empty(0, dtype=torch.long),
            }

        input_ids = [b["input_ids"] for b in batch]
        attention_mask = [b["attention_mask"] for b in batch]
        images = [b["images"] for b in batch]  # list of list-of-images
        action_label = torch.stack([b["action_label"] for b in batch]).long()

        max_len = min(max(len(x) for x in input_ids), self.max_length)

        def left_pad_and_trunc(x, pad_value):
            if len(x) > max_len:
                x = x[-max_len:]  # keep the tail (closest to decision)
            return torch.nn.functional.pad(x, (max_len - len(x), 0), value=pad_value)

        input_ids = torch.stack([left_pad_and_trunc(x, self.tokenizer.pad_token_id) for x in input_ids]).long()
        attention_mask = torch.stack([left_pad_and_trunc(x, 0) for x in attention_mask]).long()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "images": images,
            "action_label": action_label,
        }

'''
class ActionCollator:
    def __init__(self, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        batch = [b for b in batch if b is not None]
        if not batch:
            return {"input_ids": torch.empty(0), "attention_mask": torch.empty(0), "images": [], "action_label": torch.empty(0)}

        input_ids = [b["input_ids"] for b in batch]
        attention_mask = [b["attention_mask"] for b in batch]
        images = [b["images"] for b in batch]  # list of list-of-images
        action_label = torch.stack([b["action_label"] for b in batch])

        # discard too long
        filtered = []
        for ids, am, img, lab in zip(input_ids, attention_mask, images, action_label):
            if len(ids) <= self.max_length:
                filtered.append((ids, am, img, lab))
        if not filtered:
            return {"input_ids": torch.empty(0), "attention_mask": torch.empty(0), "images": [], "action_label": torch.empty(0)}

        input_ids, attention_mask, images, action_label = zip(*filtered)
        action_label = torch.stack(list(action_label))

        max_len = max(len(x) for x in input_ids)
        max_len = min(max_len, self.max_length)

        def left_pad(x, pad_value):
            if len(x) > max_len:
                x = x[-max_len:]
            return torch.nn.functional.pad(x, (max_len - len(x), 0), value=pad_value)

        input_ids = torch.stack([left_pad(x, self.tokenizer.pad_token_id) for x in input_ids])
        attention_mask = torch.stack([left_pad(x, 0) for x in attention_mask])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "images": list(images),
            "action_label": action_label,
        }'''