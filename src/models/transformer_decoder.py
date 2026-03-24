from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TransformerCaptionDecoder(nn.Module):
    """Transformer decoder for near-future captioning.

    Keeps the same external behavior as the previous LSTM decoder:
    - forward(context, captions) returns logits [B, SeqLen, vocab]
    - internally uses teacher forcing on captions[:, :-1]
    - prepends a projected context token so output length == input caption length

    Cross-attention memory is the same projected context token (length=1).
    """

    def __init__(
        self,
        *,
        context_dim: int,
        vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_len: int = 128,
        pad_token_id: Optional[int] = None,
    ):
        super().__init__()

        self.vocab_size = int(vocab_size)
        self.d_model = int(d_model)
        self.pad_token_id = pad_token_id

        self.embed = nn.Embedding(self.vocab_size, self.d_model)
        self.context_projection = nn.Linear(int(context_dim), self.d_model)

        self.positional = PositionalEncoding(self.d_model, dropout=float(dropout), max_len=int(max_len))

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=int(nhead),
            dim_feedforward=int(dim_feedforward),
            dropout=float(dropout),
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=int(num_layers))

        self.out = nn.Linear(self.d_model, self.vocab_size)

    @staticmethod
    def _causal_mask(size: int, device) -> torch.Tensor:
        # True where positions should be masked
        return torch.triu(torch.ones(size, size, device=device, dtype=torch.bool), diagonal=1)

    def forward(self, context: torch.Tensor, captions: torch.Tensor) -> torch.Tensor:
        """Args:
        context:  [B, context_dim]
        captions: [B, MaxLen]
        Returns:
        logits:   [B, SeqLen, vocab]
        """
        # Teacher forcing: use all but last token
        # We will prepend context token -> output length equals captions length
        token_ids = captions[:, :-1]

        ctx = self.context_projection(context).unsqueeze(1)  # [B, 1, D]
        tok = self.embed(token_ids) * math.sqrt(self.d_model)  # [B, L-1, D]
        tgt = torch.cat([ctx, tok], dim=1)  # [B, L, D]
        tgt = self.positional(tgt)

        # Memory for cross-attention: just the context token
        memory = ctx  # [B, 1, D]

        tgt_mask = self._causal_mask(tgt.size(1), tgt.device)

        tgt_key_padding_mask = None
        if self.pad_token_id is not None:
            # captions padding only affects tok part; context token is never padding
            pad = token_ids.eq(int(self.pad_token_id))
            # prepend False for ctx token
            tgt_key_padding_mask = torch.cat(
                [torch.zeros(pad.size(0), 1, device=pad.device, dtype=torch.bool), pad], dim=1
            )

        h = self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=None,
        )

        logits = self.out(h)  # [B, L, vocab]
        return logits
