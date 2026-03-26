import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from src.custom_tokenizer import CustomTokenizer
from tqdm import tqdm
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

# Import các module chúng ta đã viết
from src.config import Config
from src.dataset import DrivingRiskDataset
from src.models.full_model import DrivingRiskModel

class FocalLoss(nn.Module):
    """
    Focal Loss: Trị tận gốc bệnh học vẹt (Class Imbalance).
    Bóp nghẹt loss của những từ dễ (văn mẫu), phóng to loss của những từ hiếm (biển báo, xe đỗ).
    """
    def __init__(self, gamma=2.0, ignore_index=-100, label_smoothing=0.1):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(
            ignore_index=ignore_index, 
            reduction='none', 
            label_smoothing=label_smoothing
        )

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()
    
def train():
    # --- 1. THIẾT LẬP MÔI TRƯỜNG & LOGGING ---
    device = Config.DEVICE
    print(f"Dang su dung thiet bi: {device}")
    
    os.makedirs(os.path.dirname(Config.MODEL_SAVE_PATH), exist_ok=True)

    # Khởi tạo file log để ghi nhận giá trị qua từng Epoch (THÊM CỘT MAE)
    log_file = os.path.join(os.path.dirname(Config.MODEL_SAVE_PATH), "training_log.csv")
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("Epoch,Train_Loss,Val_Loss,Motion_MSE_Val,Motion_MAE_Val,Caption_Loss_Val\n")

    # --- 2. CHUẨN BỊ VÀ CHIA DỮ LIỆU ---
    print("Dang tai va chia du lieu (Train 80% / Val 10% / Test 10%)...")
    
    tokenizer = CustomTokenizer(vocab_path=Config.VOCAB_PATH, max_len=30)
    
    transform = transforms.Compose([
        transforms.Resize(Config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    full_df = pd.read_csv(Config.TRAIN_CSV)
    
    train_df, temp_df = train_test_split(full_df, test_size=0.15, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42)
    
    print(f"So luong Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    test_csv_path = os.path.join(os.path.dirname(Config.TRAIN_CSV), "test_data.csv")
    test_df.to_csv(test_csv_path, index=False)
    print(f"Da luu tap Test ra file rieng: {test_csv_path}")

    # Khoi tao Dataset
    train_dataset = DrivingRiskDataset(
        csv_file=Config.TRAIN_CSV, images_root=Config.IMAGES_ROOT,
        telemetry_root=Config.TELEMETRY_ROOT, tokenizer=tokenizer,
        transform=transform, max_frames=Config.MAX_FRAMES, future_steps=Config.FUTURE_STEPS
    )
    train_dataset.data = train_df 
    
    val_dataset = DrivingRiskDataset(
        csv_file=Config.TRAIN_CSV, images_root=Config.IMAGES_ROOT,
        telemetry_root=Config.TELEMETRY_ROOT, tokenizer=tokenizer,
        transform=transform, max_frames=Config.MAX_FRAMES, future_steps=Config.FUTURE_STEPS
    )
    val_dataset.data = val_df 
     
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=3, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=3, pin_memory=True) 
    
    # --- 3. KHỞI TẠO MODEL ---
    print("Dang khoi tao Model...")
    model = DrivingRiskModel(Config, vocab_size=len(tokenizer)).to(device)
    
    pretrain_path = os.path.join(os.path.dirname(__file__), "saved_models", "cnn_pretrained.pth")

    if os.path.exists(pretrain_path):
        print(f"Phat hien file pretrained CNN: {pretrain_path}")
        model.encoder.load_pretrained_cnn(pretrain_path)
    else:
        print(f"CẢNH BÁO: Không tìm thấy {pretrain_path}! Mô hình sẽ train CNN từ đầu.")
    
    # --- 4. CẤU HÌNH HUẤN LUYỆN ---
    criterion_caption = FocalLoss(gamma=4.0, ignore_index=tokenizer.pad_token_id, label_smoothing=0.1)
    
    # Kẹp thêm L1Loss để đo lường MAE cho nhánh vật lý
    criterion_motion_mse = nn.MSELoss() 
    criterion_motion_mae = nn.L1Loss()  

    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-5)

    best_val_loss = float('inf')
    patience = 5 
    epochs_no_improve = 0 

    # --- 5. VÒNG LẶP HUẤN LUYỆN ---
    print("BAT DAU HUAN LUYEN...")
    
    for epoch in range(Config.NUM_EPOCHS):
        # ==========================================
        #                 BƯỚC TRAIN
        # ==========================================
        model.train()
        total_train_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.NUM_EPOCHS} [Train]")
        
        for batch in loop:
            images = batch['video'].to(device)         
            sensors = batch['sensor'].to(device)      
            future_targets = batch['future_motion'].to(device) 
            captions = batch['caption'].to(device)     

            vocab_outputs, future_preds = model(images, sensors, captions)

            # 1. Loss Vật lý (Tính cả MSE để train, MAE để soi)
            loss_motion_mse = criterion_motion_mse(future_preds, future_targets)
            loss_motion_mae = criterion_motion_mae(future_preds, future_targets)
            
            # 2. Loss Văn bản
            vocab_size = len(tokenizer)
            output_flat = vocab_outputs.view(-1, vocab_size)    
            target_flat = captions.contiguous().view(-1)        
            
            loss_cap = criterion_caption(output_flat, target_flat)
            
            # 3. Tổng hợp Loss (Vẫn dùng MSE cho backprop)
            loss = (Config.LOSS_CAPTION_WEIGHT * loss_cap) + (Config.LOSS_MOTION_WEIGHT * loss_motion_mse)

            # Học ngược
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            # Hiện MAE lên thanh tiến độ
            loop.set_postfix(loss=loss.item(), mse=loss_motion_mse.item(), mae=loss_motion_mae.item(), cap=loss_cap.item())
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        # ==========================================
        #             BƯỚC VAL (THI THỬ)
        # ==========================================
        model.eval() 
        total_val_loss = 0
        total_val_motion_mse = 0
        total_val_motion_mae = 0
        total_val_cap_loss = 0
        
        with torch.no_grad(): 
            for batch in val_loader:
                images = batch['video'].to(device)
                sensors = batch['sensor'].to(device)
                future_targets = batch['future_motion'].to(device)
                captions = batch['caption'].to(device)

                vocab_outputs, future_preds = model(images, sensors, captions)

                # Tính riêng từng loại Loss
                loss_motion_mse = criterion_motion_mse(future_preds, future_targets)
                loss_motion_mae = criterion_motion_mae(future_preds, future_targets)
                
                output_flat = vocab_outputs.view(-1, len(tokenizer))
                target_flat = captions.contiguous().view(-1)  
                loss_cap = criterion_caption(output_flat, target_flat)
                
                loss = (Config.LOSS_CAPTION_WEIGHT * loss_cap) + (Config.LOSS_MOTION_WEIGHT * loss_motion_mse)
                
                total_val_loss += loss.item()
                total_val_motion_mse += loss_motion_mse.item()
                total_val_motion_mae += loss_motion_mae.item()
                total_val_cap_loss += loss_cap.item()
                
        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_motion_mse = total_val_motion_mse / len(val_loader)
        avg_val_motion_mae = total_val_motion_mae / len(val_loader)
        avg_val_cap_loss = total_val_cap_loss / len(val_loader)
        
        print(f"\n=> TỔNG KẾT EPOCH {epoch+1}:")
        print(f"   [Train] Tổng Loss: {avg_train_loss:.4f}")
        print(f"   [Val]   Tổng Loss: {avg_val_loss:.4f} | Motion MSE: {avg_val_motion_mse:.4f} | Motion MAE: {avg_val_motion_mae:.4f} | Caption Loss: {avg_val_cap_loss:.4f}\n")
        
        # Ghi vào file log CSV (Thêm cột MAE)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"{epoch+1},{avg_train_loss:.4f},{avg_val_loss:.4f},{avg_val_motion_mse:.4f},{avg_val_motion_mae:.4f},{avg_val_cap_loss:.4f}\n")
        
        # KIỂM TRA EARLY STOPPING & LƯU MODEL
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0 
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"=> Đã lưu Model xuất sắc nhất (Val Loss giảm xuống: {best_val_loss:.4f})")
        else:
            epochs_no_improve += 1
            print(f"=> Val Loss không giảm. Cảnh báo Early Stopping: {epochs_no_improve}/{patience}")
            
            if epochs_no_improve >= patience:
                print(f"=> Dừng huấn luyện sớm (Early Stopping) tại Epoch {epoch+1} để tránh Overfitting!")
                break 

if __name__ == "__main__":
    train()