import argparse
import os
import matplotlib.pyplot as plt
import pandas as pd

def plot_pretrain_curves(log_csv: str, output_path: str):
    """Vẽ biểu đồ cho bước Pre-train CNN (MSE và MAE)"""
    if not os.path.exists(log_csv):
        print(f"Bỏ qua: Không tìm thấy file {log_csv}")
        return

    df = pd.read_csv(log_csv)
    required_cols = ["Epoch", "Train_MSE", "Train_MAE", "Val_MSE", "Val_MAE"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"Lỗi format cột pretrain: Thiếu {missing}")
        return

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=180)

    # Plot 1: MSE Curve
    axes[0].plot(df["Epoch"], df["Train_MSE"], marker="o", linewidth=2.2, color="#1f77b4", label="Train MSE")
    axes[0].plot(df["Epoch"], df["Val_MSE"], marker="s", linewidth=2.2, color="#d62728", label="Val MSE")
    axes[0].set_title("CNN Pre-train MSE (Loss)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE")
    axes[0].legend(frameon=True)

    # Plot 2: MAE Curve
    axes[1].plot(df["Epoch"], df["Train_MAE"], marker="^", linewidth=2.2, color="#2ca02c", label="Train MAE")
    axes[1].plot(df["Epoch"], df["Val_MAE"], marker="D", linewidth=2.2, color="#ff7f0e", label="Val MAE")
    axes[1].set_title("CNN Pre-train MAE ", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAE")
    axes[1].legend(frameon=True)

    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle("Pre-train CNN Learning Curves", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Đã lưu biểu đồ Pre-train tại: {output_path}")

def plot_full_model_curves(log_csv: str, output_path: str):
    """Vẽ biểu đồ cho Mô hình Hybrid đầy đủ (Thêm MAE)"""
    if not os.path.exists(log_csv):
        print(f"Bỏ qua: Không tìm thấy file {log_csv}")
        return

    df = pd.read_csv(log_csv)
    # Cập nhật tên cột theo chuẩn file train.py mới nhất
    required_cols = ["Epoch", "Train_Loss", "Val_Loss", "Motion_MSE_Val", "Motion_MAE_Val", "Caption_Loss_Val"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"Lỗi format cột full model: Thiếu {missing}")
        return

    plt.style.use("seaborn-v0_8-whitegrid")
    # Tăng thành 3 biểu đồ (1x3) để tách MAE ra cho đẹp
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=180)

    # Plot 1: Tổng Loss
    axes[0].plot(df["Epoch"], df["Train_Loss"], marker="o", linewidth=2.2, color="#1f77b4", label="Train Total Loss")
    axes[0].plot(df["Epoch"], df["Val_Loss"], marker="s", linewidth=2.2, color="#d62728", label="Val Total Loss")
    axes[0].set_title("1. Tổng Loss (Train vs Val)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend(frameon=True)

    # Plot 2: Thành phần Loss ở tập Val (Motion MSE & Caption)
    axes[1].plot(df["Epoch"], df["Motion_MSE_Val"], marker="^", linewidth=2.2, color="#9467bd", label="Motion MSE (Val)")
    axes[1].plot(df["Epoch"], df["Caption_Loss_Val"], marker="v", linewidth=2.2, color="#e377c2", label="Caption Loss (Val)")
    axes[1].set_title("2. Chi tiết Val Loss (MSE & Caption)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend(frameon=True)

    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle("Driving Risk Hybrid Model Training Metrics", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Đã lưu biểu đồ Full Model tại: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vẽ biểu đồ từ các file log CSV")
    parser.add_argument("--save-dir", type=str, default="saved_models", help="Thư mục chứa file csv")
    args = parser.parse_args()

    # Đường dẫn file CSV
    pretrain_csv = os.path.join(args.save_dir, "pretrain_log.csv")
    full_csv = os.path.join(args.save_dir, "training_log.csv")

    # Đường dẫn file ảnh Output
    pretrain_out = os.path.join(args.save_dir, "pretrain_learning_curve.png")
    full_out = os.path.join(args.save_dir, "hybrid_learning_curve.png")

    print(f"Đang quét thư mục '{args.save_dir}' để vẽ biểu đồ...")
    plot_pretrain_curves(pretrain_csv, pretrain_out)
    plot_full_model_curves(full_csv, full_out)