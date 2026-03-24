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
    def __init__(self, gamma=2.0, ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        # Dùng reduction='none' để lấy loss của từng từ riêng lẻ
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')

    def forward(self, inputs, targets):
        # Tính Cross Entropy cơ bản
        ce_loss = self.ce(inputs, targets)
        
        # Lấy xác suất dự đoán (pt)
        pt = torch.exp(-ce_loss)
        
        # Ép trọng số Focal: Từ nào đoán càng đúng (pt cao) -> (1-pt) càng nhỏ -> Loss bị triệt tiêu
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        return focal_loss.mean()
    
def train():
    # --- 1. THIẾT LẬP MÔI TRƯỜNG & LOGGING ---
    device = Config.DEVICE
    print(f"Dang su dung thiet bi: {device}")
    
    os.makedirs(os.path.dirname(Config.MODEL_SAVE_PATH), exist_ok=True)

    # Khởi tạo file log để ghi nhận giá trị qua từng Epoch
    log_file = os.path.join(os.path.dirname(Config.MODEL_SAVE_PATH), "training_log.csv")
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("Epoch,Train_Loss,Val_Loss,Motion_Loss_Val,Caption_Loss_Val\n")

    # --- 2. CHUẨN BỊ VÀ CHIA DỮ LIỆU (TRAIN/VAL/TEST) ---
    print("Dang tai va chia du lieu (Train 80% / Val 10% / Test 10%)...")
    
    tokenizer = CustomTokenizer( vocab_path =Config.VOCAB_PATH, max_len=30)
    
    transform = transforms.Compose([
        transforms.Resize(Config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    full_df = pd.read_csv(Config.TRAIN_CSV)
    
    # Chia lan 1: Lay 80% cho Train, 20% cho tam (temp)
    train_df, temp_df = train_test_split(full_df, test_size=0.20, random_state=42)
    
    # Chia lan 2: Chia doi 20% temp thanh 10% Val va 10% Test
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42)
    
    print(f"So luong Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # Luu tap Test ra file rieng de danh cho buoc Evaluate (Tinh BLEU/CIDEr sau nay)
    test_csv_path = os.path.join(os.path.dirname(Config.TRAIN_CSV), "test_data.csv")
    test_df.to_csv(test_csv_path, index=False)
    print(f"Da luu tap Test ra file rieng: {test_csv_path}")

    # Khoi tao Dataset
    train_dataset = DrivingRiskDataset(
        csv_file=Config.TRAIN_CSV,
        images_root=Config.IMAGES_ROOT,
        telemetry_root=Config.TELEMETRY_ROOT,
        tokenizer=tokenizer,
        transform=transform,
        max_frames=Config.MAX_FRAMES,
        future_steps=Config.FUTURE_STEPS
    )
    train_dataset.data = train_df 
    
    val_dataset = DrivingRiskDataset(
        csv_file=Config.TRAIN_CSV,
        images_root=Config.IMAGES_ROOT,
        telemetry_root=Config.TELEMETRY_ROOT,
        tokenizer=tokenizer,
        transform=transform,
        max_frames=Config.MAX_FRAMES,
        future_steps=Config.FUTURE_STEPS
    )
    val_dataset.data = val_df 
     
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=3,pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=3,pin_memory=True) 
    # --- 3. KHỞI TẠO MODEL ---
    print("Dang khoi tao Model...")
    model = DrivingRiskModel(Config, vocab_size=len(tokenizer)).to(device)
    
    # [QUAN TRỌNG] Nạp trọng số Pre-train cho nhánh CNN
    pretrain_path = os.path.join(os.path.dirname(__file__), "saved_models", "cnn_pretrained.pth")

    if os.path.exists(pretrain_path):
        print(f"Phat hien file pretrained CNN: {pretrain_path}")
        # Truy cập vào encoder bên trong DrivingRiskModel để gọi hàm load
        model.encoder.load_pretrained_cnn(pretrain_path)
    else:
        print(f"CẢNH BÁO: Không tìm thấy {pretrain_path}! Mô hình sẽ train CNN từ đầu.")
    
    # --- 4. CẤU HÌNH HUẤN LUYỆN ---
    criterion_caption = FocalLoss(gamma=2.0, ignore_index=tokenizer.pad_token_id, label_smoothing=0.1)
    criterion_motion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, weight_decay= 1e-5)

    best_val_loss = float('inf')

    # Cấu hình Early Stopping
    patience = 5  # Số Epoch tối đa cho phép mô hình không tiến bộ
    epochs_no_improve = 0  # Bộ đếm số lần dậm chân tại chỗ

    # --- 5. VÒNG LẶP HUẤN LUYỆN (TRAINING LOOP) ---
    print("BAT DAU HUAN LUYEN...")
    
    for epoch in range(Config.NUM_EPOCHS):
        #               BUOC TRAIN
        # ==========================================
        model.train()
        total_train_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.NUM_EPOCHS} [Train]")
        
        for batch in loop:
            images = batch['video'].to(device)         # [Batch, 16, 3, 160, 90]
            sensors = batch['sensor'].to(device)      # [Batch, 5, 3] (Speed, Accel, Course)
            future_targets = batch['future_motion'].to(device) # [Batch, 5, 2] 
            captions = batch['caption'].to(device)     # [Batch, 30]

            vocab_outputs, future_preds = model(images, sensors, captions)

            # 1. Loss Vat ly
            loss_motion = criterion_motion(future_preds, future_targets)
            
            # 2. Loss Van ban
            # vocab_outputs: [B, 30, vocab_size]
            # captions: [B, 30] <- Target KHÔNG cắt [:, 1:] vì decoder đã concat context vào đầu
            vocab_size = len(tokenizer)
            output_flat = vocab_outputs.view(-1, vocab_size)    # [B*30, vocab_size]
            target_flat = captions.contiguous().view(-1)        # [B*30]
            
            loss_cap = criterion_caption(output_flat, target_flat)
            
            # 3. Tong hop Loss
            loss = loss_cap + (1.0 * loss_motion)

            # Hoc nguoc
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            loop.set_postfix(loss=loss.item(), motion=loss_motion.item(), cap=loss_cap.item())
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        # ==========================================
        #          BUOC THI THU (VALIDATION)
        # ==========================================
        model.eval() 
        total_val_loss = 0
        total_val_motion_loss = 0
        total_val_cap_loss = 0
        
        with torch.no_grad(): 
            for batch in val_loader:
                images = batch['video'].to(device)
                sensors = batch['sensor'].to(device)
                future_targets = batch['future_motion'].to(device)
                captions = batch['caption'].to(device)

                vocab_outputs, future_preds = model(images, sensors, captions)

                loss_motion = criterion_motion(future_preds, future_targets)
                
                output_flat = vocab_outputs.view(-1, len(tokenizer))
                target_flat = captions.contiguous().view(-1)  # Không cắt [:, 1:]
                loss_cap = criterion_caption(output_flat, target_flat)
                
                loss = loss_cap + (1.0 * loss_motion)
                
                total_val_loss += loss.item()
                total_val_motion_loss += loss_motion.item()
                total_val_cap_loss += loss_cap.item()
                
        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_motion_loss = total_val_motion_loss / len(val_loader)
        avg_val_cap_loss = total_val_cap_loss / len(val_loader)
        
        print(f"Ket qua Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # Ghi vao file log CSV
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"{epoch+1},{avg_train_loss:.4f},{avg_val_loss:.4f},{avg_val_motion_loss:.4f},{avg_val_cap_loss:.4f}\n")
        
        #    KIỂM TRA EARLY STOPPING & LƯU MODEL
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0 # Trả bộ đếm về 0 vì model vừa tiến bộ
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"Da luu Model xuat sac nhat (Val Loss giam xuong: {best_val_loss:.4f})")
        else:
            epochs_no_improve += 1
            print(f"Val Loss khong giam. Canh bao Early Stopping: {epochs_no_improve}/{patience}")
            
            # Nếu dậm chân tại chỗ quá giới hạn patience -> Dừng 
            if epochs_no_improve >= patience:
                print(f"Dung huan luyen som (Early Stopping) tai Epoch {epoch+1} de tranh Overfitting!")
                break 

if __name__ == "__main__":
    train()