from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence


def _normalize_text(text: str) -> str:
    text = (text or "").strip().lower()
    # keep simple punctuation as tokens by spacing them
    text = re.sub(r"([.,!?;:()\[\]\"'])", r" \1 ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@dataclass
class SimpleVocabTokenizer:
    token_to_id: Dict[str, int]
    id_to_token: List[str]

    pad_token: str = "<pad>"
    bos_token: str = "<bos>"
    eos_token: str = "<eos>"
    unk_token: str = "<unk>"

    @property
    def pad_token_id(self) -> int:
        return int(self.token_to_id[self.pad_token])

    @property
    def cls_token_id(self) -> int:
        # reuse naming to fit current code (start token)
        return int(self.token_to_id[self.bos_token])

    @property
    def sep_token_id(self) -> int:
        # reuse naming to fit current code (end token)
        return int(self.token_to_id[self.eos_token])

    @property
    def unk_token_id(self) -> int:
        return int(self.token_to_id[self.unk_token])

    def __len__(self) -> int:
        return len(self.id_to_token)

    def encode(self, text: str, *, max_length: int = 30, add_special_tokens: bool = True) -> List[int]:
        text = _normalize_text(text)
        tokens = text.split() if text else []

        ids: List[int] = []
        if add_special_tokens:
            ids.append(self.cls_token_id)

        for tok in tokens:
            ids.append(int(self.token_to_id.get(tok, self.unk_token_id)))

        if add_special_tokens:
            ids.append(self.sep_token_id)

        # pad / truncate
        ids = ids[:max_length]
        if len(ids) < max_length:
            ids = ids + [self.pad_token_id] * (max_length - len(ids))
        return ids

    def decode(self, ids: Sequence[int], *, skip_special_tokens: bool = True) -> str:
        out_tokens: List[str] = []
        for i in ids:
            i = int(i)
            if i < 0 or i >= len(self.id_to_token):
                tok = self.unk_token
            else:
                tok = self.id_to_token[i]

            if skip_special_tokens and tok in {self.pad_token, self.bos_token, self.eos_token}:
                continue
            out_tokens.append(tok)

        # re-join with simple de-spacing punctuation
        text = " ".join(out_tokens).strip()
        text = re.sub(r"\s+([.,!?;:()\]\"])", r"\1", text)
        text = re.sub(r"\(\s+", "(", text)
        text = re.sub(r"\s+\)", ")", text)
        return text.strip()

    def __call__(
        self,
        text: str,
        *,
        padding: str = "max_length",
        truncation: bool = True,
        max_length: int = 30,
        return_tensors: Optional[str] = None,
    ):
        # minimal HF-like interface
        ids = self.encode(text, max_length=max_length, add_special_tokens=True)
        if return_tensors == "pt":
            import torch

            return {"input_ids": torch.tensor([ids], dtype=torch.long)}
        return {"input_ids": ids}

    @staticmethod
    def build_from_texts(
        texts: Iterable[str],
        *,
        vocab_size: int = 4000,
        min_freq: int = 1,
    ) -> "SimpleVocabTokenizer":
        from collections import Counter

        counter: Counter[str] = Counter()
        for t in texts:
            t = _normalize_text(str(t) if t is not None else "")
            if not t:
                continue
            counter.update(t.split())

        # reserve special tokens
        specials = ["<pad>", "<bos>", "<eos>", "<unk>"]
        words = [w for w, c in counter.most_common() if c >= int(min_freq)]

        # keep top-k
        words = words[: max(int(vocab_size) - len(specials), 0)]

        id_to_token = specials + words
        token_to_id = {t: i for i, t in enumerate(id_to_token)}
        return SimpleVocabTokenizer(token_to_id=token_to_id, id_to_token=id_to_token)

    def save(self, path: str) -> None:
        payload = {
            "id_to_token": self.id_to_token,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    @staticmethod
    def load(path: str) -> "SimpleVocabTokenizer":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        id_to_token = list(payload["id_to_token"])
        token_to_id = {t: i for i, t in enumerate(id_to_token)}
        return SimpleVocabTokenizer(token_to_id=token_to_id, id_to_token=id_to_token)
