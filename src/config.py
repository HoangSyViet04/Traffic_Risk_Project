import torch
import os

basedir  = os.path.dirname(os.path.dirname(__file__))
class Config:
    # Presets:
    # - "paper": replicate Mori et al. (Near-Future captioning) baseline settings as closely as possible.
    # - "improved": current enhanced training setup.
    PRESET = "paper"

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

    # Dropout (paper dùng cấu hình đơn giản; set về 0 trong preset paper)
    ENCODER_LSTM_DROPOUT = 0.3
    DECODER_DROPOUT = 0.3
    ACTION_DROPOUT = 0.3

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

    # Init mode
    # - None: keep PyTorch defaults
    # - "xavier_paper": Xavier/orthogonal init similar to paper setting
    INIT_MODE = None

    # Thiết bị (tự động chọn GPU nếu có)
    DEVICE =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # SAVE 
    MODEL_SAVE_PATH = 'saved_models/best_model.pth'


# ---- Apply preset overrides (class-level) ----
def _apply_preset():
    preset = str(getattr(Config, "PRESET", "improved")).lower().strip()
    if preset != "paper":
        return

    # Paper: n=5 frames, 30 epochs, batch=50
    Config.MAX_FRAMES = 5
    Config.NUM_EPOCHS = 30

    # Paper: LSTM decoder
    Config.DECODER_TYPE = "lstm"

    # Paper dictionary size: 1290 (we add 4 special tokens)
    Config.TOKENIZER_TYPE = "simple"
    Config.SIMPLE_VOCAB_SIZE = 1290 + 4

    # Paper training: Adam, no weight decay, no label smoothing, no scheduler tricks
    Config.OPTIMIZER = "adam"
    Config.WEIGHT_DECAY = 0.0
    Config.LABEL_SMOOTHING = 0.0
    Config.USE_LR_SCHEDULER = False
    Config.USE_EARLY_STOPPING = False

    # Keep CNN frozen (use pretrained feature extractor)
    Config.UNFREEZE_CNN_EPOCH = None

    # Disable imbalance reweighting for paper comparability
    Config.USE_TOKEN_WEIGHTED_LOSS = False
    Config.USE_WEIGHTED_SAMPLER = False

    # Dropout not described in paper experiments: set to 0
    Config.ENCODER_LSTM_DROPOUT = 0.0
    Config.DECODER_DROPOUT = 0.0
    Config.ACTION_DROPOUT = 0.0

    # Xavier init (paper mentions Xavier)
    Config.INIT_MODE = "xavier_paper"


_apply_preset()