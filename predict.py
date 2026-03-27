import argparse
import os

import torch
from torchvision import transforms
from transformers import AutoTokenizer

from src.config import Config
from src.dataset import DrivingRiskDataset
from src.models.full_model import DrivingRiskModel


def generate_caption_and_motion(model, tokenizer, images, sensors, device, max_len=30):
    model.eval()

    with torch.no_grad():
        context = model.encoder(images, sensors)
        future_flat = model.action_head(context)
        future_pred = model.action_head.reshape_prediction(future_flat)
        decoder_context = torch.cat((context, future_flat), dim=1)

    start_token = tokenizer.cls_token_id
    end_token = tokenizer.sep_token_id
    generated_ids = [start_token]

    for _ in range(max_len - 1):
        # Decoder nội bộ luôn gọi captions[:, :-1] nên cần thêm 1 dummy token
        # ở cuối để offset đúng vị trí. Dummy PAD bị cắt bỏ trước khi vào LSTM
        # nên không ảnh hưởng gì đến hidden state. output[-1] lúc đó đúng là
        # vị trí dự đoán token tiếp theo dựa trên generated_ids[-1].
        padded_input = torch.cat([
            torch.tensor([generated_ids], dtype=torch.long, device=device),
            torch.zeros(1, 1, dtype=torch.long, device=device),   # dummy PAD, bị slice trước khi vào LSTM
        ], dim=1)

        with torch.no_grad():
            vocab_outputs = model.decoder(decoder_context, padded_input)
            next_token_id = int(vocab_outputs[0, -1, :].argmax().item())

        if next_token_id == end_token:
            break

        generated_ids.append(next_token_id)

    pred_caption = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return pred_caption, future_pred.squeeze(0)


def denormalize_future_motion(pred_motion):
    # Dataset normalizes speed by 30 and course by 360.
    out = []
    for step_idx in range(pred_motion.shape[0]):
        speed = float(pred_motion[step_idx, 0].item() * 30.0)
        course = float(pred_motion[step_idx, 1].item() * 360.0)
        out.append([round(speed, 3), round(course, 3)])
    return out


def run_single_prediction(args):
    device = Config.DEVICE
    print(f"Device: {device}")

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
    if not os.path.exists(args.test_csv):
        raise FileNotFoundError(f"Test CSV not found: {args.test_csv}")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    transform = transforms.Compose(
        [
            transforms.Resize(Config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = DrivingRiskDataset(
        csv_file=args.test_csv,
        images_root=Config.IMAGES_ROOT,
        telemetry_root=Config.TELEMETRY_ROOT,
        tokenizer=tokenizer,
        transform=transform,
        max_frames=Config.MAX_FRAMES,
        future_steps=Config.FUTURE_STEPS,
    )

    if len(dataset) == 0:
        raise ValueError("Dataset is empty")
    if args.index < 0 or args.index >= len(dataset):
        raise IndexError(f"index out of range. Valid range: [0, {len(dataset)-1}]")

    sample = dataset[args.index]

    model = DrivingRiskModel(Config, vocab_size=len(tokenizer)).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    images = sample["video"].unsqueeze(0).to(device)
    sensors = sample["sensor"].unsqueeze(0).to(device)

    pred_caption, pred_motion = generate_caption_and_motion(model, tokenizer, images, sensors, device)
    pred_motion_values = denormalize_future_motion(pred_motion.cpu())

    # Ground truth: future_motion đã được normalize trong dataset (speed/30, course/360)
    gt_motion_values = denormalize_future_motion(sample["future_motion"])
    gt_caption = tokenizer.decode(sample["caption"], skip_special_tokens=True).strip()

    print("\n" + "=" * 60)
    print(f"  Sample index : {args.index}")
    print("=" * 60)

    print("\n[GROUND TRUTH]")
    print(f"  Hanh dong that (Speed, Course) :")
    for i, (s, c) in enumerate(gt_motion_values, 1):
        print(f"    Buoc {i}: Speed = {s:>8.3f} m/s  |  Course = {c:>8.3f} deg")
    print(f"  Caption that  : {gt_caption}")

    print("\n[MODEL PREDICTION]")
    print(f"  Du doan hanh dong (Speed, Course) :")
    for i, (s, c) in enumerate(pred_motion_values, 1):
        print(f"    Buoc {i}: Speed = {s:>8.3f} m/s  |  Course = {c:>8.3f} deg")
    print(f"  Canh bao rui ro (Caption): {pred_caption}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run single-sample inference with trained model")
    parser.add_argument("--model-path", type=str, default=Config.MODEL_SAVE_PATH)
    parser.add_argument("--test-csv", type=str, default=os.path.join(os.path.dirname(Config.TRAIN_CSV), "test_data.csv"))
    parser.add_argument("--index", type=int, default=0, help="Index of sample in test_data.csv")
    args = parser.parse_args()

    run_single_prediction(args)
