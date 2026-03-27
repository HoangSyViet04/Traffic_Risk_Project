import argparse
import os

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot learning curves from training_log.csv")
    parser.add_argument("--log-csv", type=str, default=os.path.join("saved_models", "training_log.csv"))
    parser.add_argument("--output", type=str, default="learning_curve.png")
    args = parser.parse_args()

    plot_learning_curves(args.log_csv, args.output)
