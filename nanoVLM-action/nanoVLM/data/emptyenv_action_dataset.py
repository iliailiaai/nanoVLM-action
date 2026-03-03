# data/emptyenv_action_dataset.py
import os
import json
from typing import Dict, Any, List, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image

from data.processors import get_image_processor, get_tokenizer, get_image_string
from models.config import VLMConfig


ACTION2ID = {"left": 0, "right": 1, "forward": 2}

DEFAULT_PROMPT = (
    "You are controlling an agent in a grid world. "
    "Choose the next action to reach the green goal. "
    "Valid actions: left, right, forward. "
    "Answer with the best next action."
)


class EmptyEnvActionDataset(Dataset):
    """
    Reads JSONL where each line is:
      {"image": "images/....png", "action": "left|right|forward", ...}

    Outputs:
      images: processed_images (list[tensor])
      input_ids: LongTensor [T]
      attention_mask: LongTensor [T]
      action_label: LongTensor scalar
    """

    def __init__(self, jsonl_path: str, images_root: str, vlm_cfg: VLMConfig, max_samples, prompt: str = DEFAULT_PROMPT):
        self.jsonl_path = jsonl_path
        self.images_root = images_root
        self.prompt = prompt

        self.tokenizer = get_tokenizer(vlm_cfg.lm_tokenizer, vlm_cfg.vlm_extra_tokens, vlm_cfg.lm_chat_template)
        self.image_processor = get_image_processor(vlm_cfg.max_img_size, vlm_cfg.vit_img_size, vlm_cfg.resize_to_max_side_len)
        self.mp_image_token_length = vlm_cfg.mp_image_token_length

        # load lines
        self.samples: List[Dict[str, Any]] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i == max_samples: 
                    break

                line = line.strip()
                if not line:
                    continue
                self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def _process_image(self, img: Image.Image):
        if img.mode != "RGB":
            img = img.convert("RGB")

        processed_image, splitted_image_count = self.image_processor(img)

        # if tokenizer has no global_image_token but processor produced it, drop it
        if not hasattr(self.tokenizer, "global_image_token"):
            # heuristic copied from nanoVLM dataset code
            if splitted_image_count[0] * splitted_image_count[1] == len(processed_image) - 1:
                processed_image = processed_image[1:]

        return processed_image, splitted_image_count

    def __getitem__(self, idx: int):
        item = self.samples[idx]

        rel_path = item["image"]
        action_txt = item["action"].strip().lower()
        if action_txt not in ACTION2ID:
            raise ValueError(f"Unknown action: {action_txt}")

        img_path = os.path.join(self.images_root, rel_path)
        img = Image.open(img_path)

        processed_image, splitted_image_count = self._process_image(img)

        # Build a single-turn chat prompt (no history)
        # IMPORTANT: prompt always in English
        messages = [{"role": "user", "content": self.prompt}]

        image_string = get_image_string(self.tokenizer, [splitted_image_count], self.mp_image_token_length)
        messages[0]["content"] = image_string + messages[0]["content"]

        conv = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_special_tokens=False,
            return_dict=True,
        )

        input_ids = torch.tensor(conv["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(conv["attention_mask"], dtype=torch.long)
        action_label = torch.tensor(ACTION2ID[action_txt], dtype=torch.long)

        return {
            "images": [processed_image],  # keep list-of-images convention
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "action_label": action_label,
        }