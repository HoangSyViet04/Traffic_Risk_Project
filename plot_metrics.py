import argparse
import os
import re

import matplotlib.pyplot as plt
import pandas as pd


def plot_learning_curves(log_csv: str, output_path: str):
    if not os.path.exists(log_csv):
        raise FileNotFoundError(f"training log not found: {log_csv}")

    df = pd.read_csv(log_csv)
    required_cols = ["Epoch", "Train_Loss", "Val_Loss", "Motion_Loss_Val", "Caption_Loss_Val"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in log CSV: {missing}")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=180)

    # Plot 1: Main learning curve
    axes[0].plot(df["Epoch"], df["Train_Loss"], marker="o", linewidth=2.2, color="#1f77b4", label="Train Loss")
    axes[0].plot(df["Epoch"], df["Val_Loss"], marker="s", linewidth=2.2, color="#d62728", label="Val Loss")
    axes[0].set_title("Learning Curve: Train vs Validation", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend(frameon=True)

    # Plot 2: Validation loss details
    axes[1].plot(
        df["Epoch"],
        df["Motion_Loss_Val"],
        marker="^",
        linewidth=2.2,
        color="#2ca02c",
        label="Motion Loss (Val)",
    )
    axes[1].plot(
        df["Epoch"],
        df["Caption_Loss_Val"],
        marker="D",
        linewidth=2.2,
        color="#ff7f0e",
        label="Caption Loss (Val)",
    )
    axes[1].set_title("Validation Detail: Motion vs Caption", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend(frameon=True)

    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle("Driving Risk Model Training Metrics", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved learning curve image to: {output_path}")


def plot_pretrain_curves(log_csv: str, output_path: str):
    if not os.path.exists(log_csv):
        raise FileNotFoundError(f"pretrain log not found: {log_csv}")

    df = pd.read_csv(log_csv)
    required_cols = ["Epoch", "Train_MSE", "Val_MSE"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in pretrain log CSV: {missing}")

    plt.style.use("seaborn-v0_8-whitegrid")
    has_mae = "Train_MAE" in df.columns and "Val_MAE" in df.columns
    if has_mae:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=180)

        # Use different color palettes for MSE vs MAE plots.
        mse_train_color = "#1f77b4"  # blue
        mse_val_color = "#d62728"    # red
        mae_train_color = "#2ca02c"  # green
        mae_val_color = "#9467bd"    # purple

        axes[0].plot(
            df["Epoch"],
            df["Train_MSE"],
            marker="o",
            linewidth=2.2,
            color=mse_train_color,
            label="Train MSE",
        )
        axes[0].plot(
            df["Epoch"],
            df["Val_MSE"],
            marker="s",
            linewidth=2.2,
            color=mse_val_color,
            label="Val MSE",
        )
        axes[0].set_title("Pretrain CNN: MSE (Train vs Val)", fontsize=13, fontweight="bold")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("MSE")
        axes[0].legend(frameon=True)

        axes[1].plot(
            df["Epoch"],
            df["Train_MAE"],
            marker="^",
            linewidth=2.2,
            color=mae_train_color,
            label="Train MAE",
        )
        axes[1].plot(
            df["Epoch"],
            df["Val_MAE"],
            marker="D",
            linewidth=2.2,
            color=mae_val_color,
            label="Val MAE",
        )
        axes[1].set_title("Pretrain CNN: MAE (Train vs Val)", fontsize=13, fontweight="bold")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("MAE")
        axes[1].legend(frameon=True)

        # Keep the original auto-scaling behavior; only apply grid styling.
        for ax in axes:
            ax.grid(True, linestyle="--", alpha=0.4)

        fig.suptitle("Pretrain CNN Metrics", fontsize=15, fontweight="bold")
        fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    else:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=180)
        ax.plot(df["Epoch"], df["Train_MSE"], marker="o", linewidth=2.2, color="#1f77b4", label="Train MSE")
        ax.plot(df["Epoch"], df["Val_MSE"], marker="s", linewidth=2.2, color="#d62728", label="Val MSE")
        ax.set_title("Pretrain CNN: MSE (Train vs Val)", fontsize=13, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE")
        ax.legend(frameon=True)
        ax.grid(True, linestyle="--", alpha=0.4)
        fig.tight_layout()

    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved pretrain curve image to: {output_path}")


def parse_pretrain_text_log(log_txt: str) -> pd.DataFrame:
    """Parse a console-style pretrain log into a DataFrame.

    Supports lines like:
      Epoch [1/20] - Train MSE: 0.089177 | Val MSE: 0.073085
      Epoch [1/20] - Train MSE: ... | Train MAE: ... | Val MSE: ... | Val MAE: ...
    """
    if not os.path.exists(log_txt):
        raise FileNotFoundError(f"pretrain text log not found: {log_txt}")

    pattern_full = re.compile(
        r"Epoch\s*\[(?P<epoch>\d+)\/\d+\]\s*-\s*"
        r"Train\s*MSE:\s*(?P<train_mse>[0-9]*\.?[0-9]+)\s*\|\s*"
        r"Train\s*MAE:\s*(?P<train_mae>[0-9]*\.?[0-9]+)\s*\|\s*"
        r"Val\s*MSE:\s*(?P<val_mse>[0-9]*\.?[0-9]+)\s*\|\s*"
        r"Val\s*MAE:\s*(?P<val_mae>[0-9]*\.?[0-9]+)"
    )
    pattern_mse_only = re.compile(
        r"Epoch\s*\[(?P<epoch>\d+)\/\d+\]\s*-\s*"
        r"Train\s*MSE:\s*(?P<train_mse>[0-9]*\.?[0-9]+)\s*\|\s*"
        r"Val\s*MSE:\s*(?P<val_mse>[0-9]*\.?[0-9]+)"
    )

    rows = []
    with open(log_txt, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            m = pattern_full.search(line)
            if m:
                rows.append(
                    {
                        "Epoch": int(m.group("epoch")),
                        "Train_MSE": float(m.group("train_mse")),
                        "Train_MAE": float(m.group("train_mae")),
                        "Val_MSE": float(m.group("val_mse")),
                        "Val_MAE": float(m.group("val_mae")),
                    }
                )
                continue

            m = pattern_mse_only.search(line)
            if m:
                rows.append(
                    {
                        "Epoch": int(m.group("epoch")),
                        "Train_MSE": float(m.group("train_mse")),
                        "Val_MSE": float(m.group("val_mse")),
                    }
                )

    if not rows:
        raise ValueError(
            "Could not parse any epochs from text log. "
            "Expected lines starting with 'Epoch [i/N] - ...'."
        )

    df = pd.DataFrame(rows).sort_values("Epoch").reset_index(drop=True)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Plot learning curves for both training (train.py) and pretrain (pretrain.py). "
            "By default, runs in auto mode and will plot whatever logs it finds."
        )
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="auto",
        choices=["auto", "full", "pretrain"],
        help=(
            "auto=plot both when logs exist; "
            "full=train.py log (Train/Val + Motion/Caption); "
            "pretrain=pretrain.py log (MSE/MAE)"
        ),
    )
    # Backward-compatible alias for single-log workflows.
    parser.add_argument(
        "--log-csv",
        type=str,
        default=None,
        help=(
            "(legacy) Path to the log CSV for the selected --mode. "
            "Prefer using --training-log-csv / --pretrain-log-csv in auto mode."
        ),
    )
    parser.add_argument(
        "--training-log-csv",
        type=str,
        default=None,
        help="Path to train.py log CSV (default: saved_models/training_log.csv).",
    )
    parser.add_argument(
        "--pretrain-log-csv",
        type=str,
        default=None,
        help="Path to pretrain.py log CSV (default: saved_models/pretrain_log.csv).",
    )
    parser.add_argument(
        "--log-txt",
        type=str,
        default=None,
        help="(pretrain/auto) Path to a console pretrain log .txt to parse (no retrain needed).",
    )
    parser.add_argument(
        "--save-csv",
        type=str,
        default=None,
        help="(pretrain/auto) Where to write parsed CSV when using --log-txt.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Output image path for single-mode runs (--mode full/pretrain). "
            "In auto mode, use --output-dir instead."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="(auto mode) Directory to save plots (default: current directory).",
    )
    args = parser.parse_args()

    def _maybe_plot_pretrain(pretrain_csv_path: str, output_path: str) -> bool:
        if not pretrain_csv_path or not os.path.exists(pretrain_csv_path):
            return False
        plot_pretrain_curves(pretrain_csv_path, output_path)
        return True

    def _maybe_plot_full(training_csv_path: str, output_path: str) -> bool:
        if not training_csv_path or not os.path.exists(training_csv_path):
            return False
        plot_learning_curves(training_csv_path, output_path)
        return True

    # Resolve inputs
    training_log_csv = args.training_log_csv or (
        args.log_csv if args.mode == "full" else os.path.join("saved_models", "training_log.csv")
    )
    pretrain_log_csv = args.pretrain_log_csv or (
        args.log_csv if args.mode == "pretrain" else os.path.join("saved_models", "pretrain_log.csv")
    )

    # Optional: parse pretrain console log into CSV
    if args.log_txt:
        df = parse_pretrain_text_log(args.log_txt)
        save_csv = args.save_csv or pretrain_log_csv
        save_dir = os.path.dirname(save_csv)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        df.to_csv(save_csv, index=False)
        pretrain_log_csv = save_csv
        print(f"Parsed pretrain text log -> {pretrain_log_csv}")

    # Plot
    if args.mode in ("full", "pretrain"):
        if args.mode == "full":
            output = args.output or "learning_curve.png"
            plot_learning_curves(training_log_csv, output)
        else:
            output = args.output or "pretrain_curve.png"
            plot_pretrain_curves(pretrain_log_csv, output)
    else:
        os.makedirs(args.output_dir, exist_ok=True)
        out_full = os.path.join(args.output_dir, "learning_curve.png")
        out_pretrain = os.path.join(args.output_dir, "pretrain_curve.png")

        did_any = False
        did_any = _maybe_plot_full(training_log_csv, out_full) or did_any
        did_any = _maybe_plot_pretrain(pretrain_log_csv, out_pretrain) or did_any

        if not did_any:
            raise FileNotFoundError(
                "No log files found to plot. "
                "Expected one of: saved_models/training_log.csv or saved_models/pretrain_log.csv. "
                "If you already have a pretrain console log, pass --log-txt <path>."
            )
