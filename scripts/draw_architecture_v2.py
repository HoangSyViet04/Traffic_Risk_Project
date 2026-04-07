"""
Vẽ sơ đồ kiến trúc Near-Future Driving Risk Model v2 (cải tiến).
Tự động tính param counts từ model thật.

Kiến trúc v2:
  - CNN 5 lớp tự build + GAP (giữ nguyên từ pretrain)
  - Image Projection: Linear(64→256) + ReLU + Dropout
  - Encoder LSTM: input=259 (256+3), hidden=512, 2 layers, dropout + LayerNorm
  - Action Regressor: MLP thuần 512→256→128→10 (bỏ LSTM)
  - Transformer Decoder: 4-layer, 8-head, d=256, ff=1024, dropout=0.2
  - CustomTokenizer vocab ~1199
"""

import argparse, os, sys
from dataclasses import dataclass, field
from typing import List, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import (
    Circle, FancyArrowPatch, FancyBboxPatch, Rectangle,
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import Config
from src.models.full_model import DrivingRiskModel
from src.custom_tokenizer import CustomTokenizer


# ─────────────────────── stats ───────────────────────
@dataclass
class Stats:
    vocab_size: int
    total: int
    # encoder
    encoder: int
    cnn_total: int
    cnn_blocks: List[Tuple[str, int]] = field(default_factory=list)
    projection: int = 0
    enc_lstm_total: int = 0
    enc_lstm_l0: int = 0
    enc_lstm_l1: int = 0
    enc_layernorm: int = 0
    fusion_input_dim: int = 0
    projection_dim: int = 0
    # action
    action: int = 0
    action_fc1: int = 0
    action_fc2: int = 0
    action_fc3: int = 0
    # decoder
    decoder: int = 0
    dec_embed: int = 0
    dec_ctx_proj: int = 0
    dec_mem_proj: int = 0
    dec_transformer: int = 0
    dec_linear: int = 0
    dec_pos_enc: int = 0
    dec_num_layers: int = 0
    dec_nhead: int = 0
    dec_ff_dim: int = 0


def nparams(m):
    return sum(p.numel() for p in m.parameters())


def compute_stats(vocab_path: str) -> Stats:
    tok = CustomTokenizer(vocab_path)
    vocab_size = len(tok)
    model = DrivingRiskModel(Config, vocab_size=vocab_size)

    enc = model.encoder
    act = model.action_head
    dec = model.decoder

    # --- CNN blocks ---
    blocks = []
    block_id = 1
    current_label = ""
    current_params = 0
    for layer in enc.cnn.children():
        p = nparams(layer)
        cls = layer.__class__.__name__
        if cls == "Conv2d":
            if current_label:
                blocks.append((current_label, current_params))
            current_label = f"Conv({layer.in_channels}→{layer.out_channels})"
            current_params = p
            block_id += 1
        elif cls == "AdaptiveAvgPool2d":
            if current_label:
                blocks.append((current_label, current_params))
            blocks.append(("GAP", p))
            current_label = ""
            current_params = 0
        else:
            current_params += p
    if current_label:
        blocks.append((current_label, current_params))

    # --- Image Projection ---
    projection_params = nparams(enc.image_projection)

    # --- Encoder LSTM ---
    enc_lstm = enc.lstm
    l0 = (enc_lstm.weight_ih_l0.numel() + enc_lstm.weight_hh_l0.numel()
          + enc_lstm.bias_ih_l0.numel() + enc_lstm.bias_hh_l0.numel())
    l1 = (enc_lstm.weight_ih_l1.numel() + enc_lstm.weight_hh_l1.numel()
          + enc_lstm.bias_ih_l1.numel() + enc_lstm.bias_hh_l1.numel())

    # --- Action MLP layers ---
    mlp_layers = [m for m in act.mlp if hasattr(m, 'weight') and isinstance(m, __import__('torch').nn.Linear)]
    fc_params = [nparams(l) for l in mlp_layers]

    # --- Decoder ---
    # Get decoder config from the TransformerDecoderLayer
    dl = dec.transformer_decoder.layers[0]
    dec_nhead = dl.self_attn.num_heads
    dec_ff = dl.linear1.out_features
    dec_nlayers = len(dec.transformer_decoder.layers)

    return Stats(
        vocab_size=vocab_size,
        total=nparams(model),
        encoder=nparams(enc),
        cnn_total=nparams(enc.cnn),
        cnn_blocks=blocks,
        projection=projection_params,
        enc_lstm_total=l0 + l1,
        enc_lstm_l0=l0,
        enc_lstm_l1=l1,
        enc_layernorm=nparams(enc.context_norm),
        fusion_input_dim=enc.projection_dim + Config.SENSOR_DIM,
        projection_dim=enc.projection_dim,
        action=nparams(act),
        action_fc1=fc_params[0] if len(fc_params) > 0 else 0,
        action_fc2=fc_params[1] if len(fc_params) > 1 else 0,
        action_fc3=fc_params[2] if len(fc_params) > 2 else 0,
        decoder=nparams(dec),
        dec_embed=nparams(dec.embed),
        dec_ctx_proj=nparams(dec.context_projection),
        dec_mem_proj=nparams(dec.memory_projection),
        dec_transformer=nparams(dec.transformer_decoder),
        dec_linear=nparams(dec.linear),
        dec_pos_enc=nparams(dec.pos_encoder),
        dec_num_layers=dec_nlayers,
        dec_nhead=dec_nhead,
        dec_ff_dim=dec_ff,
    )


def fmt(n): return f"{n:,}"
def mb(n): return n * 4 / (1024 ** 2)


# ─────────────────────── drawing helpers ─────────────────────
def arrow(ax, x1, y1, x2, y2, color="#4b5563", lw=1.3, ms=10):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
        arrowstyle="-|>", mutation_scale=ms, linewidth=lw, color=color, zorder=4))

def box(ax, x, y, w, h, fc, ec, lw=1.0, zorder=2):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, linewidth=lw, zorder=zorder))

def dashed_box(ax, x, y, w, h, ec, lw=1.1):
    ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=ec,
        linewidth=lw, linestyle=(0, (4, 3)), zorder=1))

def rounded_box(ax, x, y, w, h, fc, ec, lw=1.1):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=2))

def txt(ax, x, y, s, **kw):
    defaults = dict(ha="center", va="center", fontsize=8.5)
    defaults.update(kw)
    ax.text(x, y, s, **defaults)


# ─────────────────────── main draw ───────────────────────────
def draw(s: Stats, out_path: str):
    fig, ax = plt.subplots(figsize=(22, 11), dpi=260)
    ax.set_xlim(0, 220)
    ax.set_ylim(-6, 102)
    ax.axis("off")
    ax.add_patch(Rectangle((0, -6), 220, 108, facecolor="#f8fafc", edgecolor="none"))

    # ════════════ TITLE ════════════
    txt(ax, 110, 98, "Near-Future Driving Risk Model (v2 — Improved)", fontsize=17, fontweight="bold")
    txt(ax, 110, 95, f"Total params: {fmt(s.total)} | fp32 memory: {mb(s.total):.2f} MB | vocab: {s.vocab_size}",
        fontsize=10, color="#334155")

    # ════════════ INPUT IMAGES ════════════
    for i in range(6):
        ax.add_patch(Rectangle((4 + i * 0.8, 62 + i * 0.8), 20, 13,
            facecolor="none", edgecolor="#3b82f6", linewidth=0.9, zorder=2))
    txt(ax, 15, 59, "Input images 3@90×160", fontsize=9)
    txt(ax, 15, 56.5, "Observed sequence (T=16)", fontsize=7.8, color="#475569")

    # ════════════ MOTION INFO ════════════
    for i in range(5):
        ax.add_patch(Rectangle((6 + i * 0.8, 36 + i * 0.8), 16, 9,
            facecolor="#e5e7eb", edgecolor="#6b7280", linewidth=0.8, zorder=2))
    txt(ax, 15, 40, "Speed / Accel / Course", fontsize=8)
    txt(ax, 15, 34.5, "Motion Information", fontsize=10)

    # ════════════ CNN 5-LAYER TRUNK ════════════
    cnn_x0, cnn_y0 = 30, 54
    cnn_w, cnn_h = 34, 32
    dashed_box(ax, cnn_x0, cnn_y0, cnn_w, cnn_h, ec="#64748b")

    bar_labels = ["C1\n3→16", "P", "C2\n16→32", "P", "C3\n32→48", "P",
                  "C4\n48→64", "C5\n64→64", "GAP"]
    bar_colors = ["#bfdbfe", "#fde68a", "#bfdbfe", "#fde68a", "#bfdbfe", "#fde68a",
                  "#bfdbfe", "#bfdbfe", "#d9f99d"]
    bar_ec     = ["#60a5fa", "#f59e0b", "#60a5fa", "#f59e0b", "#60a5fa", "#f59e0b",
                  "#60a5fa", "#60a5fa", "#65a30d"]
    bx = cnn_x0 + 1.2
    bar_heights = [27, 25, 23, 21, 19, 17, 15, 13, 11]
    for i, (lbl, bh) in enumerate(zip(bar_labels, bar_heights)):
        by = cnn_y0 + 3 + (27 - bh) / 2
        bw = 3.2
        box(ax, bx, by, bw, bh, fc=bar_colors[i], ec=bar_ec[i], lw=0.85)
        txt(ax, bx + bw / 2, by + bh / 2, lbl, fontsize=5.8, rotation=90)
        bx += bw + 0.3

    txt(ax, cnn_x0 + cnn_w / 2, cnn_y0 - 1.8, "CNN-5 Trunk (Pretrained)",
        fontsize=10.5, fontweight="bold")
    txt(ax, cnn_x0 + cnn_w / 2, cnn_y0 - 4, f"Params: {fmt(s.cnn_total)}",
        fontsize=8.5, color="#1d4ed8")
    txt(ax, cnn_x0 + cnn_w / 2, cnn_y0 - 6,
        "Output: [B, 64, 1, 1] → flatten [B, 64]", fontsize=7.2, color="#334155")

    # ════════════ IMAGE PROJECTION ════════════
    proj_x, proj_y = 68, 64
    box(ax, proj_x, proj_y, 14, 8, fc="#e0e7ff", ec="#6366f1", lw=1.0)
    txt(ax, proj_x + 7, proj_y + 5.5, "Projection", fontsize=8, fontweight="bold")
    txt(ax, proj_x + 7, proj_y + 3.5, "64→256", fontsize=7.5)
    txt(ax, proj_x + 7, proj_y + 1.5, f"+ReLU+Drop", fontsize=6.5, color="#4b5563")
    txt(ax, proj_x + 7, proj_y - 1.5, f"{fmt(s.projection)} params", fontsize=7, color="#6366f1")

    # ════════════ ENCODER LSTM BLOCK ════════════
    enc_x0, enc_y0, enc_w, enc_h = 88, 58, 32, 22
    dashed_box(ax, enc_x0, enc_y0, enc_w, enc_h, ec="#60a5fa")

    for r in range(2):
        for c in range(3):
            rx = enc_x0 + 2.5 + c * 9.5
            ry = enc_y0 + 12 - r * 7.5
            box(ax, rx, ry, 8, 4.5, fc="#d9f99d", ec="#65a30d", lw=0.9)
            txt(ax, rx + 4, ry + 2.25, "LSTM", fontsize=7.5)
            if c < 2:
                arrow(ax, rx + 8.1, ry + 2.25, rx + 9.3, ry + 2.25, lw=0.6, ms=6)
            if r == 0:
                arrow(ax, rx + 4, ry, rx + 4, ry - 3.0, lw=0.6, ms=6)

    # LayerNorm badge
    box(ax, enc_x0 + enc_w - 8, enc_y0 + 1, 8, 3, fc="#fef3c7", ec="#f59e0b", lw=0.8)
    txt(ax, enc_x0 + enc_w - 4, enc_y0 + 2.5, "LayerNorm", fontsize=6.5)

    txt(ax, enc_x0 + enc_w / 2, enc_y0 - 1.5, f"Encoder Context {Config.HIDDEN_SIZE}",
        fontsize=11, fontweight="bold")
    txt(ax, enc_x0 + enc_w / 2, enc_y0 - 3.5,
        f"LSTM 2-layer: input={s.fusion_input_dim}, hidden={Config.HIDDEN_SIZE}, dropout=0.3",
        fontsize=7.5, color="#166534")
    txt(ax, enc_x0 + enc_w / 2, enc_y0 - 5.5,
        f"L1 {fmt(s.enc_lstm_l0)} | L2 {fmt(s.enc_lstm_l1)} | Total: {fmt(s.enc_lstm_total)}",
        fontsize=7, color="#166534")
    txt(ax, enc_x0 + enc_w / 2, enc_y0 - 7.5,
        f"Encoder total: {fmt(s.encoder)} ({s.encoder / s.total * 100:.1f}%)",
        fontsize=7.5, color="#1d4ed8")

    # ════════════ CONCAT CIRCLE ════════════
    concat_cx, concat_cy = 130, 44
    ax.add_patch(Circle((concat_cx, concat_cy), 2.5,
        facecolor="#f3f4f6", edgecolor="black", linewidth=1.0, zorder=3))
    txt(ax, concat_cx, concat_cy, ";", fontsize=16, fontweight="bold")

    # ════════════ ACTION REGRESSOR (MLP) ════════════
    act_x0, act_y0, act_w, act_h = 140, 54, 56, 13
    ax.add_patch(Rectangle((act_x0, act_y0), act_w, act_h,
        fill=False, edgecolor="#374151", linewidth=1.1, zorder=2))

    # FC1
    fc1_x = act_x0 + 2
    box(ax, fc1_x, act_y0 + 3.5, 14, 7, fc="#dbeafe", ec="#3b82f6", lw=0.95)
    txt(ax, fc1_x + 7, act_y0 + 8.8, "FC1 + ReLU", fontsize=7)
    txt(ax, fc1_x + 7, act_y0 + 7, f"512→256", fontsize=6.8)
    txt(ax, fc1_x + 7, act_y0 + 5.2, f"+Dropout(0.3)", fontsize=6, color="#6b7280")
    txt(ax, fc1_x + 7, act_y0 + 3.8, fmt(s.action_fc1), fontsize=6.5, color="#166534")

    # FC2
    fc2_x = fc1_x + 16
    box(ax, fc2_x, act_y0 + 3.5, 14, 7, fc="#eff6ff", ec="#3b82f6", lw=0.9)
    txt(ax, fc2_x + 7, act_y0 + 8.8, "FC2 + ReLU", fontsize=7)
    txt(ax, fc2_x + 7, act_y0 + 7, f"256→128", fontsize=6.8)
    txt(ax, fc2_x + 7, act_y0 + 5.2, f"+Dropout(0.2)", fontsize=6, color="#6b7280")
    txt(ax, fc2_x + 7, act_y0 + 3.8, fmt(s.action_fc2), fontsize=6.5, color="#166534")

    # FC3
    fc3_x = fc2_x + 16
    box(ax, fc3_x, act_y0 + 3.5, 10, 7, fc="#eff6ff", ec="#3b82f6", lw=0.9)
    txt(ax, fc3_x + 5, act_y0 + 8.3, "FC3", fontsize=7.5)
    txt(ax, fc3_x + 5, act_y0 + 6.5, "128→10", fontsize=6.8)
    txt(ax, fc3_x + 5, act_y0 + 4.5, fmt(s.action_fc3), fontsize=6.5, color="#166534")

    # reshape
    box(ax, act_x0 + 2, act_y0 + 0.8, 52, 2.3, fc="#eff6ff", ec="#3b82f6", lw=0.7)
    txt(ax, act_x0 + 28, act_y0 + 1.95, "future_flat [B,10] → reshape [B,5,2]", fontsize=6.8)

    # internal arrows
    arrow(ax, fc1_x + 14.1, act_y0 + 7, fc2_x, act_y0 + 7, lw=0.7, ms=6)
    arrow(ax, fc2_x + 14.1, act_y0 + 7, fc3_x, act_y0 + 7, lw=0.7, ms=6)

    # Action labels
    txt(ax, act_x0 + act_w / 2, act_y0 - 1.5, "Action Regressor (MLP)", fontsize=12, fontweight="bold")
    txt(ax, act_x0 + act_w / 2, act_y0 - 3.5,
        f"FC1 {fmt(s.action_fc1)} | FC2 {fmt(s.action_fc2)} | FC3 {fmt(s.action_fc3)}",
        fontsize=7.5, color="#166534")
    txt(ax, act_x0 + act_w / 2, act_y0 - 5.5,
        f"Total: {fmt(s.action)} ({s.action / s.total * 100:.1f}%)",
        fontsize=8, color="#166534")

    # ════════════ TRANSFORMER DECODER ════════════
    dec_x0, dec_y0, dec_w, dec_h = 86, 4, 110, 32
    dashed_box(ax, dec_x0, dec_y0, dec_w, dec_h, ec="#f87171")

    # ── Memory Projection ──
    box(ax, dec_x0 + 2, dec_y0 + 24, 16, 5, fc="#fef3c7", ec="#f59e0b", lw=0.9)
    txt(ax, dec_x0 + 10, dec_y0 + 27.5, "Memory Proj", fontsize=7)
    txt(ax, dec_x0 + 10, dec_y0 + 25.5, f"{Config.HIDDEN_SIZE}→256", fontsize=6.5)

    # ── Context Projection ──
    box(ax, dec_x0 + 2, dec_y0 + 16.5, 16, 5, fc="#fef3c7", ec="#f59e0b", lw=0.9)
    txt(ax, dec_x0 + 10, dec_y0 + 20, "Ctx Proj", fontsize=7)
    context_dim = Config.HIDDEN_SIZE + Config.FUTURE_STEPS * 2
    txt(ax, dec_x0 + 10, dec_y0 + 18, f"{context_dim}→256", fontsize=6.5)

    # ── Embedding ──
    box(ax, dec_x0 + 21, dec_y0 + 24, 14, 5, fc="#e0e7ff", ec="#6366f1", lw=0.9)
    txt(ax, dec_x0 + 28, dec_y0 + 27.5, "Embedding", fontsize=7)
    txt(ax, dec_x0 + 28, dec_y0 + 25.5, f"{s.vocab_size}→256", fontsize=6.5)

    # ── Positional Encoding ──
    box(ax, dec_x0 + 21, dec_y0 + 16.5, 14, 5, fc="#e0e7ff", ec="#6366f1", lw=0.9)
    txt(ax, dec_x0 + 28, dec_y0 + 20, "Pos Encoding", fontsize=7)
    txt(ax, dec_x0 + 28, dec_y0 + 18, "d=256", fontsize=6.5)

    # ── Transformer Decoder Layers ──
    trf_x = dec_x0 + 38
    layer_h = 5.5
    for i in range(s.dec_num_layers):
        ly = dec_y0 + 8 + i * (layer_h + 1.5)
        fc_color = "#fce7f3" if i % 2 == 0 else "#fdf2f8"
        box(ax, trf_x, ly, 34, layer_h, fc=fc_color, ec="#db2777", lw=1.0)
        txt(ax, trf_x + 17, ly + 3.8, f"TransformerDecoderLayer {i+1}", fontsize=7, fontweight="bold")
        txt(ax, trf_x + 17, ly + 1.8, f"{s.dec_nhead}-head Attn + FFN({s.dec_ff_dim})", fontsize=6.5)
        txt(ax, trf_x + 17, ly + 0.5, f"d_model=256", fontsize=5.8, color="#6b7280")
        if i < s.dec_num_layers - 1:
            arrow(ax, trf_x + 17, ly + layer_h + 0.1, trf_x + 17, ly + layer_h + 1.3, lw=0.6, ms=5)

    # ── Causal Mask ──
    mask_x = trf_x + 36
    box(ax, mask_x, dec_y0 + 14, 12, 5, fc="#fee2e2", ec="#ef4444", lw=0.9)
    txt(ax, mask_x + 6, dec_y0 + 17.5, "Causal", fontsize=7)
    txt(ax, mask_x + 6, dec_y0 + 15.5, "Mask", fontsize=7)

    # ── Output Linear ──
    box(ax, mask_x, dec_y0 + 24, 12, 5, fc="#dcfce7", ec="#22c55e", lw=0.9)
    txt(ax, mask_x + 6, dec_y0 + 27.5, "Linear", fontsize=7.5)
    txt(ax, mask_x + 6, dec_y0 + 25.5, f"256→{s.vocab_size}", fontsize=6.5)

    # ── Output tokens ──
    tokens = ["BOS", "The", "car", "moves", "right", "...", "EOS"]
    tx = dec_x0 + 8
    for i, tk in enumerate(tokens):
        box(ax, tx, dec_y0 + 1, 9, 2.8, fc="#f3f4f6", ec="#fb923c", lw=0.8)
        txt(ax, tx + 4.5, dec_y0 + 2.4, tk, fontsize=6.8)
        if i < len(tokens) - 1:
            arrow(ax, tx + 9.1, dec_y0 + 2.4, tx + 10.2, dec_y0 + 2.4, lw=0.5, ms=5)
        tx += 10.5

    # Internal decoder arrows
    arrow(ax, dec_x0 + 18, dec_y0 + 26.5, dec_x0 + 21, dec_y0 + 26.5, lw=0.7, ms=6)  # mem→embed area
    arrow(ax, dec_x0 + 18, dec_y0 + 19, dec_x0 + 21, dec_y0 + 19, lw=0.7, ms=6)       # ctx→pos
    arrow(ax, dec_x0 + 35, dec_y0 + 26.5, trf_x, dec_y0 + 26.5, lw=0.7, ms=6)         # embed→trf memory
    arrow(ax, dec_x0 + 35, dec_y0 + 19, trf_x, dec_y0 + 14, lw=0.7, ms=6)             # pos→trf input
    arrow(ax, trf_x + 34, dec_y0 + 26.5, mask_x, dec_y0 + 26.5, lw=0.7, ms=6)         # trf→linear

    # Decoder labels
    txt(ax, dec_x0 + dec_w / 2, dec_y0 - 1.5,
        "Transformer Caption Decoder (4-layer, 8-head)", fontsize=14, fontweight="bold")
    txt(ax, dec_x0 + dec_w / 2, dec_y0 - 3.5,
        f"Embed {fmt(s.dec_embed)} | CtxProj {fmt(s.dec_ctx_proj)} | MemProj {fmt(s.dec_mem_proj)} "
        f"| Transformer {fmt(s.dec_transformer)} | Linear {fmt(s.dec_linear)}",
        fontsize=7.5, color="#9d174d")
    txt(ax, dec_x0 + dec_w / 2, dec_y0 - 5.5,
        f"Total: {fmt(s.decoder)} ({s.decoder / s.total * 100:.1f}%) "
        f"| Output logits [B, 30, {s.vocab_size}]",
        fontsize=7.5, color="#9d174d")

    # ════════════ SETTINGS PANEL ════════════
    sp_x, sp_y, sp_w, sp_h = 140, 69, 72, 24
    rounded_box(ax, sp_x, sp_y, sp_w, sp_h, fc="#ffffff", ec="#94a3b8")
    txt(ax, sp_x + sp_w / 2, sp_y + sp_h - 2.2, "MODEL SETTINGS (v2)",
        fontsize=10.5, fontweight="bold", color="#0f172a")

    settings = [
        f"IMAGE_SIZE: {Config.IMAGE_SIZE}",
        f"MAX_FRAMES: {Config.MAX_FRAMES}",
        f"SENSOR_DIM: {Config.SENSOR_DIM}",
        f"HIDDEN_SIZE: {Config.HIDDEN_SIZE}",
        f"EMBED_SIZE: {Config.EMBED_SIZE}",
        f"FUTURE_STEPS: {Config.FUTURE_STEPS}",
        f"CNN: 5 conv + GAP → 64-d → Projection 256-d",
        f"Encoder LSTM: input={s.fusion_input_dim}, hidden={Config.HIDDEN_SIZE}, 2-layer, drop=0.3",
        f"Action MLP: {Config.HIDDEN_SIZE}→256→128→10 (no LSTM)",
        f"Decoder: Transformer {s.dec_num_layers}-layer, {s.dec_nhead}-head, d=256, ff={s.dec_ff_dim}",
        f"Train: batch={Config.BATCH_SIZE}, epochs={Config.NUM_EPOCHS}, lr={Config.LEARNING_RATE}",
        f"Optimizer: Adam + CosineAnnealingLR + GradClip(1.0)",
    ]
    sy = sp_y + sp_h - 4.5
    for line in settings:
        txt(ax, sp_x + 2, sy, line, fontsize=7, ha="left", color="#1e293b")
        sy -= 1.7

    # ════════════ ARROWS (pipeline connections) ════════════
    # images → CNN
    arrow(ax, 25, 69, 30, 69)
    # CNN → projection
    arrow(ax, 64, 69, 68, 69)
    # projection → encoder
    arrow(ax, 82, 69, 88, 69)
    # motion → encoder
    arrow(ax, 23, 40, 88, 40)
    arrow(ax, 88, 40, 88, 62, lw=1.0, ms=8)

    # encoder → action
    arrow(ax, 120, 66, 130, 66)
    arrow(ax, 130, 66, 140, 66, lw=1.0, ms=8)

    # encoder context → concat
    arrow(ax, 120, 62, 130, 62)
    arrow(ax, 130, 62, 130, 46.8)

    # action → concat (future_flat)
    arrow(ax, 196, 60, 200, 60)
    arrow(ax, 200, 60, 200, 44)
    arrow(ax, 200, 44, 132.8, 44)

    # concat → decoder (ctx proj)
    arrow(ax, 130, 41.2, 130, 38)
    arrow(ax, 130, 38, 96, 38)
    arrow(ax, 96, 38, 96, 36.2)

    # encoder lstm_out → decoder (memory proj)
    arrow(ax, 120, 60, 124, 60)
    arrow(ax, 124, 60, 124, 38)
    arrow(ax, 124, 38, 96, 38)

    # ════════════ FOOTNOTE ════════════
    txt(ax, 110, -5,
        "Metrics auto-computed from instantiated model + CustomTokenizer. "
        "Improvements: Image Projection, LayerNorm, MLP Action, 4-layer Transformer Decoder, CosineAnnealingLR.",
        fontsize=7.5, color="#64748b")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    plt.savefig(out_path, dpi=320, bbox_inches="tight")
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Draw v2 architecture diagram")
    parser.add_argument("--out", default="docs/model_architecture_v2.png")
    parser.add_argument("--vocab", default=None)
    args = parser.parse_args()

    vocab_path = args.vocab or Config.VOCAB_PATH
    s = compute_stats(vocab_path)

    print(f"{'Vocab size':>20s} : {s.vocab_size}")
    print(f"{'Total params':>20s} : {fmt(s.total)}")
    print(f"{'  Encoder':>20s} : {fmt(s.encoder)} ({s.encoder/s.total*100:.1f}%)")
    print(f"{'    CNN':>20s} : {fmt(s.cnn_total)}")
    print(f"{'    Projection':>20s} : {fmt(s.projection)}")
    print(f"{'    LSTM':>20s} : {fmt(s.enc_lstm_total)}")
    print(f"{'    LayerNorm':>20s} : {fmt(s.enc_layernorm)}")
    print(f"{'  Action (MLP)':>20s} : {fmt(s.action)} ({s.action/s.total*100:.1f}%)")
    print(f"{'  Decoder':>20s} : {fmt(s.decoder)} ({s.decoder/s.total*100:.1f}%)")
    print(f"{'    Transformer':>20s} : {fmt(s.dec_transformer)}")
    print(f"{'    Embed':>20s} : {fmt(s.dec_embed)}")
    print(f"{'    Linear':>20s} : {fmt(s.dec_linear)}")

    draw(s, args.out)


if __name__ == "__main__":
    main()
