import torch
import os

basedir  = os.path.dirname(os.path.dirname(__file__))
class Config:
    # Đường dẫn 
    TRAIN_CSV = os.path.join(basedir, 'data', 'processed_train.csv')
    IMAGES_ROOT = os.path.join(basedir, 'data', 'images')
    TELEMETRY_ROOT = os.path.join(basedir, 'data', 'telemetry')

    # cấu hình model
    # Kích thước ảnh đầu vào 
    IMAGE_SIZE = (90, 160)

    # Số frame model sẽ nhìn (Start -> Mid)
    MAX_FRAMES = 16

    # FPS của ảnh đã extract (scripts/extract_frames.py lấy mỗi 6 frame từ video 30fps)
    FRAME_FPS = 5

    # Telemetry sampling rate heuristic
    # - "auto": suy luận dựa trên độ dài mảng locations
    # - "1hz" | "5fps" | "30fps": ép cố định
    TELEMETRY_RATE_MODE = "auto"

    # Kích thước vector nhúng (Embedding cho word)
    EMBED_SIZE = 256
    # Kích thước hidden state cho LSTM (Encoder + Decoder)
    HIDDEN_SIZE = 1024

    # Decoder type
    # - "lstm": decoder cũ
    # - "transformer": transformer decoder mới (khuyến nghị)
    DECODER_TYPE = "transformer"

    # Transformer decoder hyperparams
    TRANSFORMER_D_MODEL = 512
    TRANSFORMER_NHEAD = 8
    TRANSFORMER_NUM_LAYERS = 4
    TRANSFORMER_FF_DIM = 2048
    TRANSFORMER_DROPOUT = 0.1

    # Số lượng tham số sensor (Speed, Acceleration, Course)
    SENSOR_DIM = 3

    # Action Regressor dự đoán số bước tương lai
    FUTURE_STEPS = 5

    # --- HUẤN LUYỆN ---
    BATCH_SIZE = 50
    NUM_EPOCHS = 40
    LEARNING_RATE = 3e-4

    OPTIMIZER = "adamw"  # "adam" | "adamw"
    WEIGHT_DECAY = 0.01
    LABEL_SMOOTHING = 0.1

    # LR scheduling
    USE_LR_SCHEDULER = True
    LR_SCHEDULER_FACTOR = 0.5
    LR_SCHEDULER_PATIENCE = 2
    LR_SCHEDULER_MIN_LR = 1e-6

    # Trộn loss đa nhiệm
    MOTION_LOSS_WEIGHT = 1.0

    # Chống exploding gradients
    GRAD_CLIP_NORM = 1.0

    # AMP (tự bật khi có CUDA)
    USE_AMP = True

    # Mất cân bằng caption (word/token-level + sample-level)
    USE_TOKEN_WEIGHTED_LOSS = True
    TOKEN_WEIGHT_ALPHA = 0.5
    MAX_TOKEN_WEIGHT = 10.0
    USE_WEIGHTED_SAMPLER = True
    SAMPLE_WEIGHT_POWER = 1.0

    # Fine-tune CNN sau N epoch (None = luôn freeze)
    UNFREEZE_CNN_EPOCH = 3

    # Early stopping
    USE_EARLY_STOPPING = True
    EARLY_STOPPING_PATIENCE = 12
    EARLY_STOPPING_MIN_DELTA = 1e-4

    # Tokenizer
    # - "bert": như hiện tại
    # - "simple": vocab nhỏ, word-level (thường giúp caption metrics tốt hơn với dataset nhỏ)
    TOKENIZER_TYPE = "bert"
    SIMPLE_VOCAB_SIZE = 4000
    VOCAB_SAVE_PATH = os.path.join("saved_models", "simple_vocab.json")

    # Thiết bị (tự động chọn GPU nếu có)
    DEVICE =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # SAVE 
    MODEL_SAVE_PATH = 'saved_models/best_model.pth'