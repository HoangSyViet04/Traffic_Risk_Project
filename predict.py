import argparse
import os

import torch
from torchvision import transforms
from transformers import AutoTokenizer

from src.config import Config
from src.dataset import DrivingRiskDataset
from src.models.full_model import DrivingRiskModel
from src.simple_tokenizer import SimpleVocabTokenizer


def _load_tokenizer_for_inference():
    tok_type = str(getattr(Config, "TOKENIZER_TYPE", "bert")).lower().strip()
    if tok_type == "simple":
        vocab_path = getattr(Config, "VOCAB_SAVE_PATH", None) or getattr(Config, "VOCAB_PATH", None)
        if not vocab_path:
            vocab_path = "saved_models/simple_vocab.json"
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(
                f"Simple vocab not found at {vocab_path}. Run training once to create it."
            )
        return SimpleVocabTokenizer.load(vocab_path)
    return AutoTokenizer.from_pretrained("bert-base-uncased")


def generate_caption_and_motion_beam(
    model,
    tokenizer,
    images,
    sensors,
    device,
    *,
    max_len: int = 30,
    beam_size: int = 3,
    length_penalty: float = 0.7,
):
    model.eval()

    with torch.no_grad():
        context = model.encoder(images, sensors)
        future_flat = model.action_head(context)
        future_pred = model.action_head.reshape_prediction(future_flat)
        decoder_context_1 = torch.cat((context, future_flat), dim=1)  # [1, 1034]

    start_token = int(tokenizer.cls_token_id)
    end_token = int(tokenizer.sep_token_id)
    pad_token = int(tokenizer.pad_token_id) if tokenizer.pad_token_id is not None else None

    beams = [([start_token], 0.0, False)]

    def length_norm(score: float, length: int) -> float:
        lp = float(length_penalty)
        if lp <= 0:
            return score
        denom = ((5.0 + float(length)) / 6.0) ** lp
        return score / denom

    for _ in range(max_len - 1):
        all_candidates = []
        active = [(i, b) for i, b in enumerate(beams) if not b[2]]
        if not active:
            break

        prefix_batch = [b[0] for _, b in active]
        max_prefix_len = max(len(x) for x in prefix_batch)
        prefix_tensor = torch.full(
            (len(prefix_batch), max_prefix_len),
            fill_value=pad_token if pad_token is not None else 0,
            dtype=torch.long,
            device=device,
        )
        for row_i, seq in enumerate(prefix_batch):
            prefix_tensor[row_i, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)

        padded_input = torch.cat(
            [prefix_tensor, torch.zeros(len(prefix_batch), 1, dtype=torch.long, device=device)],
            dim=1,
        )

        decoder_context = decoder_context_1.expand(len(prefix_batch), -1)
        with torch.no_grad():
            vocab_outputs = model.decoder(decoder_context, padded_input)
            logits = vocab_outputs[:, -1, :]
            log_probs = torch.log_softmax(logits, dim=-1)
            if pad_token is not None:
                log_probs[:, pad_token] = -1e9
            topk = torch.topk(log_probs, k=max(int(beam_size), 1), dim=-1)

        for local_idx, (_beam_idx, (seq, score, _)) in enumerate(active):
            for tok_id, tok_lp in zip(topk.indices[local_idx].tolist(), topk.values[local_idx].tolist()):
                new_seq = seq + [int(tok_id)]
                new_score = float(score) + float(tok_lp)
                ended = int(tok_id) == end_token
                all_candidates.append((new_seq, new_score, ended))

        for seq, score, ended in beams:
            if ended:
                all_candidates.append((seq, score, True))

        all_candidates.sort(key=lambda x: length_norm(x[1], len(x[0])), reverse=True)
        beams = all_candidates[: max(int(beam_size), 1)]

        if all(b[2] for b in beams):
            break

    best_seq, _, _ = max(beams, key=lambda x: length_norm(x[1], len(x[0])))
    pred_caption = tokenizer.decode(best_seq, skip_special_tokens=True).strip()
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
    if isinstance(device, str):
        device = torch.device(device)
    print(f"Device: {device}")

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(
            f"Model file not found: {args.model_path}. "
            f"Train first (python train.py) or pass --model-path to an existing checkpoint."
        )
    if not os.path.exists(args.test_csv):
        raise FileNotFoundError(f"Test CSV not found: {args.test_csv}")

    tokenizer = _load_tokenizer_for_inference()
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
        frame_fps=getattr(Config, "FRAME_FPS", 5),
        telemetry_rate_mode=getattr(Config, "TELEMETRY_RATE_MODE", "auto"),
    )

    if len(dataset) == 0:
        raise ValueError("Dataset is empty")
    if args.index < 0 or args.index >= len(dataset):
        raise IndexError(f"index out of range. Valid range: [0, {len(dataset)-1}]")

    sample = dataset[args.index]

    model = DrivingRiskModel(Config, vocab_size=len(tokenizer)).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    if hasattr(model.decoder, "pad_token_id"):
        try:
            model.decoder.pad_token_id = tokenizer.pad_token_id
        except Exception:
            pass

    images = sample["video"].unsqueeze(0).to(device)
    sensors = sample["sensor"].unsqueeze(0).to(device)

    pred_caption, pred_motion = generate_caption_and_motion_beam(
        model,
        tokenizer,
        images,
        sensors,
        device,
        max_len=30,
        beam_size=int(args.beam_size),
        length_penalty=float(args.length_penalty),
    )
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
    default_model_path = getattr(Config, "MODEL_SAVE_PATH", "saved_models/best_model.pth")
    default_model_dir = os.path.dirname(default_model_path) or "saved_models"
    default_test_csv = getattr(Config, "TEST_CSV", None) or os.path.join(default_model_dir, "test_data.csv")
    parser.add_argument("--test-csv", type=str, default=default_test_csv)
    parser.add_argument("--index", type=int, default=0, help="Index of sample in test_data.csv")
    parser.add_argument("--beam-size", type=int, default=3, help="Beam size for caption decoding (1 = greedy)")
    parser.add_argument("--length-penalty", type=float, default=0.7, help="Beam search length penalty")
    args = parser.parse_args()

    run_single_prediction(args)
