import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import pandas as pd
import json
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
        alt_root = os.path.join(os.path.dirname(Config.TRAIN_CSV), "images_resized")
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


def run_pretrain(train_loader, val_loader, epochs=10, lr=1e-4, device=None, save_path="cnn_pretrained.pth"):
    """
    Huấn luyện PretrainCNN với train/val và lưu best model.

    Yêu cầu loader trả về:
        images: [B, 3, 90, 160]
        targets: [B, 2]  (Speed, Course)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PretrainCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_loss = []
    val_loss = []

    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            preds = model(images)                  # [B, 2]
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / max(1, len(train_loader))
        train_loss.append(avg_train_loss)
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                targets = targets.to(device)

                preds = model(images)
                loss = criterion(preds, targets)
                val_loss += loss.item()

        avg_val_loss = val_loss / max(1, len(val_loader))
        val_loss.append(val_loss)
        print(f"Epoch [{epoch+1}/{epochs}] - Train MSE: {avg_train_loss:.6f} | Val MSE: {avg_val_loss:.6f}")


        with open("loss_log.json", "w") as f:
            json.dump({
        "train": train_loss,
        "val": val_loss
                }, f)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print(f"Saved best pretrain model -> {save_path} (val={best_val_loss:.6f})")


if __name__ == "__main__":
    train_loader, val_loader = build_pretrain_loaders()
    run_pretrain(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=Config.NUM_EPOCHS,
        lr=Config.LEARNING_RATE,
        device=Config.DEVICE,
        save_path = os.path.join(os.path.dirname(__file__), "saved_models", "cnn_pretrained.pth")
    )
