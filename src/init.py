from __future__ import annotations

import torch
import torch.nn as nn


def init_weights_xavier_paper(module: nn.Module) -> None:
    """Initialization similar in spirit to the paper's Xavier init.

    - Linear: Xavier uniform, bias zeros
    - Embedding: Xavier uniform
    - LSTM: Xavier uniform for input weights, orthogonal for recurrent weights, bias zeros

    CNN weights are intentionally not touched here (pretrained feature extractor).
    """

    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
        return

    if isinstance(module, nn.Embedding):
        nn.init.xavier_uniform_(module.weight)
        return

    if isinstance(module, nn.LSTM):
        for name, param in module.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                nn.init.zeros_(param.data)
        return


def apply_paper_init(model: nn.Module, *, skip_cnn: bool = True) -> None:
    """Apply init to all submodules, optionally skipping CNN."""

    for name, sub in model.named_modules():
        if skip_cnn and (name.endswith("cnn") or name.startswith("encoder.cnn")):
            continue
        init_weights_xavier_paper(sub)
