import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from transformers import AutoTokenizer
from tqdm import tqdm
import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Import các module chúng ta đã viết
from src.config import Config
from src.dataset import DrivingRiskDataset
from src.models.full_model import DrivingRiskModel
from src.utils import build_imbalance_weights
from src.simple_tokenizer import SimpleVocabTokenizer


def _build_tokenizer(train_texts):
    tok_type = getattr(Config, "TOKENIZER_TYPE", "bert")
    tok_type = str(tok_type).lower().strip()

    if tok_type == "simple":
        vocab_size = int(getattr(Config, "SIMPLE_VOCAB_SIZE", 4000))
        tokenizer = SimpleVocabTokenizer.build_from_texts(train_texts, vocab_size=vocab_size)
        os.makedirs(os.path.dirname(Config.VOCAB_SAVE_PATH), exist_ok=True)
        tokenizer.save(Config.VOCAB_SAVE_PATH)
        print(f"Saved simple vocab tokenizer to: {Config.VOCAB_SAVE_PATH}")
        return tokenizer

    # default: BERT tokenizer
    return AutoTokenizer.from_pretrained("bert-base-uncased")

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

    # Build tokenizer after we have train split (avoid leaking val/test distribution)
    tokenizer = _build_tokenizer(train_df["caption"].astype(str).tolist())

    # Khoi tao Dataset
    train_dataset = DrivingRiskDataset(
        csv_file=Config.TRAIN_CSV,
        images_root=Config.IMAGES_ROOT,
        telemetry_root=Config.TELEMETRY_ROOT,
        tokenizer=tokenizer,
        transform=transform,
        max_frames=Config.MAX_FRAMES,
        future_steps=Config.FUTURE_STEPS,
        frame_fps=Config.FRAME_FPS,
        telemetry_rate_mode=Config.TELEMETRY_RATE_MODE,
    )
    train_dataset.data = train_df 
    
    val_dataset = DrivingRiskDataset(
        csv_file=Config.TRAIN_CSV,
        images_root=Config.IMAGES_ROOT,
        telemetry_root=Config.TELEMETRY_ROOT,
        tokenizer=tokenizer,
        transform=transform,
        max_frames=Config.MAX_FRAMES,
        future_steps=Config.FUTURE_STEPS,
        frame_fps=Config.FRAME_FPS,
        telemetry_rate_mode=Config.TELEMETRY_RATE_MODE,
    )
    val_dataset.data = val_df 

    # --- (OPTIONAL) XỬ LÝ MẤT CÂN BẰNG CAPTION ---
    # 1) Token-weighted loss: giảm bias về từ phổ biến
    # 2) Weighted sampler: tăng xác suất lấy sample có token hiếm
    imbalance = None
    if getattr(Config, "USE_TOKEN_WEIGHTED_LOSS", False) or getattr(Config, "USE_WEIGHTED_SAMPLER", False):
        exclude_ids = [
            tokenizer.cls_token_id,
            tokenizer.sep_token_id,
            tokenizer.unk_token_id,
        ]
        imbalance = build_imbalance_weights(
            texts=train_df["caption"].astype(str).tolist(),
            tokenizer=tokenizer,
            vocab_size=len(tokenizer),
            pad_token_id=tokenizer.pad_token_id,
            max_length=30,
            token_alpha=getattr(Config, "TOKEN_WEIGHT_ALPHA", 0.5),
            max_token_weight=getattr(Config, "MAX_TOKEN_WEIGHT", 10.0),
            sample_power=getattr(Config, "SAMPLE_WEIGHT_POWER", 1.0),
            exclude_special_token_ids=exclude_ids,
        )

    sampler = None
    if getattr(Config, "USE_WEIGHTED_SAMPLER", False) and imbalance is not None:
        sampler = WeightedRandomSampler(
            weights=imbalance.sample_weights,
            num_samples=len(imbalance.sample_weights),
            replacement=True,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=0,
    )
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0) 
    # --- 3. KHỞI TẠO MODEL ---
    print("Dang khoi tao Model...")
    model = DrivingRiskModel(Config, vocab_size=len(tokenizer)).to(device)

    # Provide PAD id to Transformer decoder if supported
    if hasattr(model.decoder, "pad_token_id"):
        try:
            model.decoder.pad_token_id = tokenizer.pad_token_id
        except Exception:
            pass
    
    # [QUAN TRỌNG] Nạp trọng số Pre-train cho nhánh CNN
    pretrain_path = os.path.join(os.path.dirname(__file__), "saved_models", "cnn_pretrained.pth")

    if os.path.exists(pretrain_path):
        print(f"Phat hien file pretrained CNN: {pretrain_path}")
        # Truy cập vào encoder bên trong DrivingRiskModel để gọi hàm load
        model.encoder.load_pretrained_cnn(pretrain_path)
    else:
        print(f"CẢNH BÁO: Không tìm thấy {pretrain_path}! Mô hình sẽ train CNN từ đầu.")
    
    # --- 4. CẤU HÌNH HUẤN LUYỆN ---
    label_smoothing = float(getattr(Config, "LABEL_SMOOTHING", 0.0))
    if label_smoothing < 0:
        label_smoothing = 0.0

    if getattr(Config, "USE_TOKEN_WEIGHTED_LOSS", False) and imbalance is not None:
        token_w = imbalance.token_weights.to(device)
        criterion_caption = nn.CrossEntropyLoss(
            ignore_index=tokenizer.pad_token_id,
            weight=token_w,
            label_smoothing=label_smoothing,
        )
    else:
        criterion_caption = nn.CrossEntropyLoss(
            ignore_index=tokenizer.pad_token_id,
            label_smoothing=label_smoothing,
        )
    criterion_motion = nn.MSELoss()

    # Optimizer
    opt_name = str(getattr(Config, "OPTIMIZER", "adamw")).lower().strip()
    wd = float(getattr(Config, "WEIGHT_DECAY", 0.0))
    if opt_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=wd)

    # Scheduler
    scheduler = None
    if bool(getattr(Config, "USE_LR_SCHEDULER", True)):
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(getattr(Config, "LR_SCHEDULER_FACTOR", 0.5)),
            patience=int(getattr(Config, "LR_SCHEDULER_PATIENCE", 2)),
            min_lr=float(getattr(Config, "LR_SCHEDULER_MIN_LR", 1e-6)),
        )

    use_amp = bool(getattr(Config, "USE_AMP", True)) and (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val_loss = float('inf')

    # Cấu hình Early Stopping
    use_early_stopping = bool(getattr(Config, "USE_EARLY_STOPPING", True))
    patience = int(getattr(Config, "EARLY_STOPPING_PATIENCE", 12))
    min_delta = float(getattr(Config, "EARLY_STOPPING_MIN_DELTA", 0.0))
    epochs_no_improve = 0  # Bộ đếm số lần dậm chân tại chỗ

    # --- 5. VÒNG LẶP HUẤN LUYỆN (TRAINING LOOP) ---
    print("BAT DAU HUAN LUYEN...")
    
    for epoch in range(Config.NUM_EPOCHS):
        # Optional: unfreeze CNN after some warmup epochs
        unfreeze_epoch = getattr(Config, "UNFREEZE_CNN_EPOCH", None)
        if unfreeze_epoch is not None and epoch == int(unfreeze_epoch):
            print(f"Unfreezing CNN at epoch {epoch+1}...")
            model.encoder.freeze_cnn = False

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

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                vocab_outputs, future_preds = model(images, sensors, captions)

                # 1. Loss Vat ly
                loss_motion = criterion_motion(future_preds, future_targets)

                # 2. Loss Van ban
                vocab_size = len(tokenizer)
                output_flat = vocab_outputs.view(-1, vocab_size)    # [B*30, vocab_size]
                target_flat = captions.contiguous().view(-1)        # [B*30]
                loss_cap = criterion_caption(output_flat, target_flat)

                # 3. Tong hop Loss
                motion_w = float(getattr(Config, "MOTION_LOSS_WEIGHT", 1.0))
                loss = loss_cap + (motion_w * loss_motion)

            scaler.scale(loss).backward()

            clip_norm = getattr(Config, "GRAD_CLIP_NORM", None)
            if clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip_norm))

            scaler.step(optimizer)
            scaler.update()

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

                with torch.cuda.amp.autocast(enabled=use_amp):
                    vocab_outputs, future_preds = model(images, sensors, captions)

                    loss_motion = criterion_motion(future_preds, future_targets)

                    output_flat = vocab_outputs.view(-1, len(tokenizer))
                    target_flat = captions.contiguous().view(-1)  # Không cắt [:, 1:]
                    loss_cap = criterion_caption(output_flat, target_flat)

                    motion_w = float(getattr(Config, "MOTION_LOSS_WEIGHT", 1.0))
                    loss = loss_cap + (motion_w * loss_motion)
                
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
        if avg_val_loss < (best_val_loss - min_delta):
            best_val_loss = avg_val_loss
            epochs_no_improve = 0 # Trả bộ đếm về 0 vì model vừa tiến bộ
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"Da luu Model xuat sac nhat (Val Loss giam xuong: {best_val_loss:.4f})")
        else:
            epochs_no_improve += 1
            print(f"Val Loss khong giam. Canh bao Early Stopping: {epochs_no_improve}/{patience}")
            
            # Nếu dậm chân tại chỗ quá giới hạn patience -> Dừng 
            if use_early_stopping and epochs_no_improve >= patience:
                print(f"Dung huan luyen som (Early Stopping) tai Epoch {epoch+1} de tranh Overfitting!")
                break 

        if scheduler is not None:
            scheduler.step(avg_val_loss)

if __name__ == "__main__":
    train()