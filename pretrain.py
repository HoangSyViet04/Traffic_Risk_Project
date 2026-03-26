import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import pandas as pd
from tqdm import tqdm  # THÊM THƯ VIỆN NÀY ĐỂ HIỆN THANH TIẾN ĐỘ

from src.config import Config
from src.pretrain_dataset import SingleFrameDataset
from src.models.pretrain_cnn import PretrainCNN


def build_pretrain_loaders(batch_size=None, val_ratio=0.1):
    """
    Tạo train/val loader riêng cho pre-train ảnh đơn.
    """
    if batch_size is None:
        batch_size = Config.BATCH_SIZE

    # Ưu tiên đường dẫn hiện có trong config, fallback nếu thư mục ảnh khác tên.
    images_root = Config.IMAGES_ROOT
    if not os.path.exists(images_root):
        alt_root = os.path.join(os.path.dirname(Config.TRAIN_CSV), "images")
        if os.path.exists(alt_root):
            images_root = alt_root

    transform = transforms.Compose(
        [
            transforms.Resize(Config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    full_df = pd.read_csv(Config.TRAIN_CSV)
    train_df, val_df = train_test_split(full_df, test_size=val_ratio, random_state=42)

    train_ds = SingleFrameDataset(
        csv_file=Config.TRAIN_CSV,
        images_root=images_root,
        telemetry_root=Config.TELEMETRY_ROOT,
        transform=transform,
        timestamp_mode="mid",
    )
    train_ds.data = train_df.reset_index(drop=True)

    val_ds = SingleFrameDataset(
        csv_file=Config.TRAIN_CSV,
        images_root=images_root,
        telemetry_root=Config.TELEMETRY_ROOT,
        transform=transform,
        timestamp_mode="mid",
    )
    val_ds.data = val_df.reset_index(drop=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader


def run_pretrain(train_loader, val_loader, epochs=20, lr=1e-4, device=None, save_path="cnn_pretrained.pth"):
    """
    Huấn luyện PretrainCNN: Loss là MSE, log thêm MAE.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PretrainCNN().to(device)
    
    # Khai báo 2 cây thước đo
    criterion_mse = nn.MSELoss() # Dùng để huấn luyện
    criterion_mae = nn.L1Loss()  # Dùng để đo lường MAE
    
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_mse = float("inf")

    for epoch in range(epochs):
        # ================= TRAIN =================
        model.train()
        train_mse_sum, train_mae_sum = 0.0, 0.0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for images, targets in train_pbar:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            preds = model(images)                  
            
            # Tính 2 loại loss
            loss_mse = criterion_mse(preds, targets)
            loss_mae = criterion_mae(preds, targets)
            
            # Backprop bằng MSE
            loss_mse.backward()
            optimizer.step()

            train_mse_sum += loss_mse.item()
            train_mae_sum += loss_mae.item()
            
            # Hiện cả MSE và MAE lên thanh chạy
            train_pbar.set_postfix({"MSE": f"{loss_mse.item():.4f}", "MAE": f"{loss_mae.item():.4f}"})

        avg_train_mse = train_mse_sum / max(1, len(train_loader))
        avg_train_mae = train_mae_sum / max(1, len(train_loader))

        # ================= VALIDATION =================
        model.eval()
        val_mse_sum, val_mae_sum = 0.0, 0.0
        
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]  ")
        with torch.no_grad():
            for images, targets in val_pbar:
                images = images.to(device)
                targets = targets.to(device)

                preds = model(images)
                
                loss_mse = criterion_mse(preds, targets)
                loss_mae = criterion_mae(preds, targets)
                
                val_mse_sum += loss_mse.item()
                val_mae_sum += loss_mae.item()
                
                val_pbar.set_postfix({"MSE": f"{loss_mse.item():.4f}", "MAE": f"{loss_mae.item():.4f}"})

        avg_val_mse = val_mse_sum / max(1, len(val_loader))
        avg_val_mae = val_mae_sum / max(1, len(val_loader))

        # Tổng kết in ra màn hình
        print(f"\n=> TỔNG KẾT EPOCH {epoch+1}:")
        print(f"   [Train] MSE: {avg_train_mse:.5f} | MAE: {avg_train_mae:.5f}")
        print(f"   [Val]   MSE: {avg_val_mse:.5f} | MAE: {avg_val_mae:.5f}")

        # Lưu model dựa trên độ tụt của Validation MSE
        if avg_val_mse < best_val_mse:
            best_val_mse = avg_val_mse
            torch.save(model.state_dict(), save_path)
            print(f"-> Đã lưu Model tốt nhất tại -> {save_path} (Val MSE={best_val_mse:.5f})\n")


if __name__ == "__main__":
    train_loader, val_loader = build_pretrain_loaders()
    print(f"Pretrain - Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Kéo thẳng đường dẫn từ Config sang để 2 file nằm cạnh nhau
    save_dir = os.path.join(Config.BASE_DIR, "saved_models")
    save_path = os.path.join(save_dir, "cnn_pretrained.pth")
    
    run_pretrain(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=Config.NUM_EPOCHS,
        lr=Config.LEARNING_RATE,
        device=Config.DEVICE,
        save_path=save_path
    )