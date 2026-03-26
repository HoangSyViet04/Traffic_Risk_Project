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
    parser = argparse.ArgumentParser(description="Plot learning curves from training_log.csv")
    parser.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=["full", "pretrain"],
        help="full=train.py log (Train/Val + Motion/Caption), pretrain=pretrain.py log (MSE/MAE)",
    )
    parser.add_argument(
        "--log-csv",
        type=str,
        default=None,
        help="Path to the log CSV. If omitted, uses a mode-specific default.",
    )
    parser.add_argument(
        "--log-txt",
        type=str,
        default=None,
        help="(pretrain mode) Path to a console log .txt to parse (no retrain needed).",
    )
    parser.add_argument(
        "--save-csv",
        type=str,
        default=None,
        help="(pretrain mode) Where to write parsed CSV when using --log-txt.",
    )
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.mode == "pretrain":
        # If user provides a text log from an older run, parse it into CSV first.
        if args.log_txt:
            df = parse_pretrain_text_log(args.log_txt)
            save_csv = args.save_csv or (args.log_csv or os.path.join("saved_models", "pretrain_log.csv"))
            os.makedirs(os.path.dirname(save_csv), exist_ok=True)
            df.to_csv(save_csv, index=False)
            log_csv = save_csv
            print(f"Parsed pretrain text log -> {log_csv}")
        else:
            log_csv = args.log_csv or os.path.join("saved_models", "pretrain_log.csv")
        output = args.output or "pretrain_curve.png"
        plot_pretrain_curves(log_csv, output)
    else:
        log_csv = args.log_csv or os.path.join("saved_models", "training_log.csv")
        output = args.output or "learning_curve.png"
        plot_learning_curves(log_csv, output)
