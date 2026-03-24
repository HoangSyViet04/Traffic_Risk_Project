from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch


@dataclass(frozen=True)
class ImbalanceWeights:
	token_weights: torch.Tensor  # [vocab_size], float32
	sample_weights: torch.Tensor  # [n_samples], float64 (for WeightedRandomSampler)


def _tokenize_ids(
	tokenizer,
	text: str,
	max_length: int,
) -> List[int]:
	encoded = tokenizer(
		text,
		truncation=True,
		max_length=max_length,
		padding=False,
		return_tensors=None,
	)
	# HF tokenizer returns dict with 'input_ids'
	ids = encoded["input_ids"]
	return list(ids)


def build_imbalance_weights(
	texts: Sequence[str],
	tokenizer,
	*,
	vocab_size: int,
	pad_token_id: Optional[int],
	max_length: int = 30,
	token_alpha: float = 0.5,
	max_token_weight: float = 10.0,
	sample_power: float = 1.0,
	exclude_special_token_ids: Optional[Iterable[int]] = None,
) -> ImbalanceWeights:
	"""Build weights to mitigate caption imbalance.

	- token_weights: inverse-frequency style weights for CrossEntropyLoss(weight=...)
	- sample_weights: per-sample weights for WeightedRandomSampler

	Notes:
	- Works with large vocab tokenizers (e.g., BERT). Only tokens observed get reweighted.
	- PAD is ignored later by loss, but we still set its weight to 0 when provided.
	"""

	if exclude_special_token_ids is None:
		exclude_special_token_ids = []
	exclude_special_token_ids = set(int(x) for x in exclude_special_token_ids)
	if pad_token_id is not None:
		exclude_special_token_ids.add(int(pad_token_id))

	counts = np.zeros(int(vocab_size), dtype=np.int64)
	for t in texts:
		ids = _tokenize_ids(tokenizer, str(t) if t is not None else "", max_length)
		for tok in ids:
			tok = int(tok)
			if tok in exclude_special_token_ids:
				continue
			if 0 <= tok < vocab_size:
				counts[tok] += 1

	nonzero = counts[counts > 0]
	token_weights = np.ones(int(vocab_size), dtype=np.float32)
	if nonzero.size > 0:
		mean_count = float(nonzero.mean())
		for tok_id, c in enumerate(counts.tolist()):
			if c <= 0:
				continue
			w = (mean_count / float(c)) ** float(token_alpha)
			if max_token_weight is not None:
				w = float(min(float(max_token_weight), float(w)))
			token_weights[tok_id] = float(w)

	if pad_token_id is not None and 0 <= int(pad_token_id) < vocab_size:
		token_weights[int(pad_token_id)] = 0.0

	token_weights_t = torch.tensor(token_weights, dtype=torch.float32)

	# Sample weights: mean token weight of the caption (excluding PAD + special tokens)
	tw_np = token_weights_t.cpu().numpy()
	sample_weights: List[float] = []
	for t in texts:
		ids = _tokenize_ids(tokenizer, str(t) if t is not None else "", max_length)
		kept = [int(x) for x in ids if int(x) not in exclude_special_token_ids and 0 <= int(x) < vocab_size]
		if not kept:
			w = 1.0
		else:
			w = float(np.mean(tw_np[kept]))
		w = float(max(w, 1e-6))
		sample_weights.append(w**float(sample_power))

	sample_weights_t = torch.tensor(sample_weights, dtype=torch.float64)
	return ImbalanceWeights(token_weights=token_weights_t, sample_weights=sample_weights_t)

