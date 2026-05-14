from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from src.training.sft import (
    normalize_conversations,
    render_chat_prompt,
    build_loss_labels,
    ROLE_MARKERS,
)


_DEFAULT_EOS_TOKEN = '<|endoftext|>'


class DPODataset(Dataset):
    """
    A dataset for Direct Preference Optimization (DPO) training.

    Expected JSONL format (each line is a dict):
    {
        "prompt": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "question"}
        ],
        "chosen": [{"role": "assistant", "content": "preferred answer"}],
        "rejected": [{"role": "assistant", "content": "rejected answer"}]
    }
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        tokenizer,
        max_length: int = 1024,
        eos_token: str | None = None,
    ) -> None:
        super().__init__()
        self.jsonl_path = Path(jsonl_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.eos_token = eos_token if eos_token is not None else _DEFAULT_EOS_TOKEN
        self.assistant_header_ids = tokenizer.encode(ROLE_MARKERS["assistant"])
        self.eos_boundary_ids = tokenizer.encode(f"{self.eos_token}\n")
        eos_token_bytes = self.eos_token.encode("utf-8")
        if eos_token_bytes not in tokenizer.vocab_to_id:
            raise ValueError(f"EOS token {self.eos_token!r} is not in the tokenizer vocabulary")
        self.pad_token_id = tokenizer.vocab_to_id[eos_token_bytes]
        self._offsets: list[int] = []

        with self.jsonl_path.open("r", encoding="utf-8") as f:
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                if line.strip():
                    self._offsets.append(offset)

        if not self._offsets:
            raise ValueError(f"No usable DPO samples found in {self.jsonl_path}")

    def __len__(self) -> int:
        return len(self._offsets)

    def _read_sample(self, index: int) -> dict[str, object]:
        with self.jsonl_path.open("r", encoding="utf-8") as f:
            f.seek(self._offsets[index])
            return json.loads(f.readline())

    def _encode_conversations(self, conversations: list[dict[str, str]]) -> tuple[list[int], list[int]]:
        rendered = render_chat_prompt(conversations, eos_token=self.eos_token, add_generation_prompt=False)
        input_ids = self.tokenizer.encode(rendered)[: self.max_length]
        input_ids += [self.pad_token_id] * (self.max_length - len(input_ids))
        labels = build_loss_labels(
            input_ids=input_ids,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            assistant_header_ids=self.assistant_header_ids,
            eos_boundary_ids=self.eos_boundary_ids,
            pad_token_id=self.pad_token_id,
        )
        return input_ids, labels

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self._read_sample(index)

        prompt = sample.get("prompt", [])
        chosen = sample.get("chosen", [])
        rejected = sample.get("rejected", [])

        if not isinstance(prompt, list) or not isinstance(chosen, list) or not isinstance(rejected, list):
            raise ValueError(f"DPO sample must contain prompt/chosen/rejected lists, got keys: {list(sample.keys())}")

        chosen_full = normalize_conversations(prompt + chosen)
        rejected_full = normalize_conversations(prompt + rejected)

        chosen_input_ids, chosen_labels = self._encode_conversations(chosen_full)
        rejected_input_ids, rejected_labels = self._encode_conversations(rejected_full)

        return {
            "chosen_input_ids": torch.tensor(chosen_input_ids, dtype=torch.long),
            "chosen_labels": torch.tensor(chosen_labels, dtype=torch.long),
            "chosen_loss_masks": torch.tensor([1 if l != -100 else 0 for l in chosen_labels], dtype=torch.long),
            "rejected_input_ids": torch.tensor(rejected_input_ids, dtype=torch.long),
            "rejected_labels": torch.tensor(rejected_labels, dtype=torch.long),
            "rejected_loss_masks": torch.tensor([1 if l != -100 else 0 for l in rejected_labels], dtype=torch.long),
        }
