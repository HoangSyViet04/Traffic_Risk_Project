import argparse
import collections
import datetime as _dt
import json
import math
import os
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from src.custom_tokenizer import CustomTokenizer

from src.config import Config
from src.dataset import DrivingRiskDataset
from src.models.full_model import DrivingRiskModel


def _safe_word_tokenize(text: str) -> List[str]:
    return text.lower().strip().split()


def _sentence_bleu4(reference: str, hypothesis: str) -> float:
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

    ref_tokens = _safe_word_tokenize(reference)
    hyp_tokens = _safe_word_tokenize(hypothesis)
    if not hyp_tokens:
        return 0.0

    return sentence_bleu(
        [ref_tokens],
        hyp_tokens,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=SmoothingFunction().method1,
    )


def _meteor_score(reference: str, hypothesis: str) -> float:
    try:
        from nltk.translate.meteor_score import meteor_score

        return meteor_score([_safe_word_tokenize(reference)], _safe_word_tokenize(hypothesis))
    except Exception:
        # Fallback when NLTK resources are unavailable: unigram F1
        ref = collections.Counter(_safe_word_tokenize(reference))
        hyp = collections.Counter(_safe_word_tokenize(hypothesis))
        if not ref or not hyp:
            return 0.0
        overlap = sum((ref & hyp).values())
        precision = overlap / max(sum(hyp.values()), 1)
        recall = overlap / max(sum(ref.values()), 1)
        if precision + recall == 0:
            return 0.0
        return (2 * precision * recall) / (precision + recall)


def _ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    if len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def _build_document_frequency(references: List[List[str]]) -> Dict[int, collections.Counter]:
    df = {n: collections.Counter() for n in range(1, 5)}
    for ref_tokens in references:
        for n in range(1, 5):
            unique_ngrams = set(_ngrams(ref_tokens, n))
            for gram in unique_ngrams:
                df[n][gram] += 1
    return df


def _tfidf_vector(tokens: List[str], n: int, df: collections.Counter, n_docs: int) -> Dict[Tuple[str, ...], float]:
    grams = _ngrams(tokens, n)
    tf = collections.Counter(grams)
    vec: Dict[Tuple[str, ...], float] = {}
    for gram, count in tf.items():
        idf = math.log((n_docs + 1.0) / (df.get(gram, 0) + 1.0))
        vec[gram] = float(count) * idf
    return vec


def _cosine_similarity(vec_a: Dict[Tuple[str, ...], float], vec_b: Dict[Tuple[str, ...], float]) -> float:
    if not vec_a or not vec_b:
        return 0.0

    dot = 0.0
    for key, val in vec_a.items():
        dot += val * vec_b.get(key, 0.0)

    norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
    norm_b = math.sqrt(sum(v * v for v in vec_b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def cider_score(references: List[str], hypotheses: List[str], sigma: float = 6.0) -> float:
    ref_tokens_list = [_safe_word_tokenize(x) for x in references]
    hyp_tokens_list = [_safe_word_tokenize(x) for x in hypotheses]

    n_docs = len(ref_tokens_list)
    if n_docs == 0:
        return 0.0

    df = _build_document_frequency(ref_tokens_list)

    sample_scores = []
    for ref_tokens, hyp_tokens in zip(ref_tokens_list, hyp_tokens_list):
        per_n_scores = []
        for n in range(1, 5):
            ref_vec = _tfidf_vector(ref_tokens, n, df[n], n_docs)
            hyp_vec = _tfidf_vector(hyp_tokens, n, df[n], n_docs)
            cos = _cosine_similarity(ref_vec, hyp_vec)

            # Gaussian penalty for length mismatch, similar to CIDEr design.
            len_penalty = math.exp(-((len(hyp_tokens) - len(ref_tokens)) ** 2) / (2 * sigma * sigma))
            per_n_scores.append(cos * len_penalty)

        sample_scores.append(10.0 * float(np.mean(per_n_scores)))

    return float(np.mean(sample_scores))


def official_cider_score_if_available(references: List[str], hypotheses: List[str]) -> Tuple[float, str]:
    """
    Returns (score, mode).
    mode = "official" when pycocoevalcap is available, otherwise "approx".
    """
    try:
        from pycocoevalcap.cider.cider import Cider

        gts = {i: [references[i]] for i in range(len(references))}
        res = {i: [hypotheses[i]] for i in range(len(hypotheses))}
        scorer = Cider()
        score, _ = scorer.compute_score(gts, res)
        return float(score), "official"
    except Exception:
        return cider_score(references, hypotheses), "approx"


def generate_caption_and_motion(
    model,
    tokenizer,
    images,
    sensors,
    device,
    max_len=30,
    beam_size: int = None,
    length_penalty_alpha: float = None,
    min_decode_len: int = None,
):
    model.eval()

    with torch.no_grad():
        # 1. Trích xuất context chung (Image + Sensor)
        context = model.encoder(images, sensors)
        
        # 2. Dự đoán tương lai
        future_flat = model.action_head(context)
        future_pred = model.action_head.reshape_prediction(future_flat)
        
        # 3. Nối thành context vector 1034-d cho Decoder
        decoder_context = torch.cat((context, future_flat), dim=1)

        # 4. GỌI HÀM BEAM SEARCH TỪ DECODER VỪA TẠO
        start_token = tokenizer.cls_token_id
        end_token = tokenizer.sep_token_id
        
        best_token_ids = model.decoder.generate_beam_search(
            context=decoder_context,
            start_token_id=start_token,
            end_token_id=end_token,
            max_len=max_len,
            beam_size=int(beam_size if beam_size is not None else getattr(Config, "BEAM_SIZE", 5)),
            length_penalty_alpha=float(
                length_penalty_alpha
                if length_penalty_alpha is not None
                else getattr(Config, "LENGTH_PENALTY_ALPHA", 0.7)
            ),
            min_len=int(min_decode_len if min_decode_len is not None else getattr(Config, "MIN_DECODE_LEN", 3)),
        )

    # 5. Dịch Token IDs thành văn bản, tự động bỏ qua các thẻ [CLS], [SEP], [PAD]
    pred_caption = tokenizer.decode(best_token_ids, skip_special_tokens=True).strip()
    
    return pred_caption, future_pred.squeeze(0)


def evaluate(args):
    device = Config.DEVICE
    print(f"Device: {device}")

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
    if not os.path.exists(args.test_csv):
        raise FileNotFoundError(f"Test CSV not found: {args.test_csv}")

    tokenizer = CustomTokenizer(vocab_path=Config.VOCAB_PATH, max_len=30)
    transform = transforms.Compose(
        [
            transforms.Resize(Config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    test_dataset = DrivingRiskDataset(
        csv_file=args.test_csv,
        images_root=Config.IMAGES_ROOT,
        telemetry_root=Config.TELEMETRY_ROOT,
        tokenizer=tokenizer,
        transform=transform,
        max_frames=Config.MAX_FRAMES,
        future_steps=Config.FUTURE_STEPS,
        sample_fps=Config.SAMPLE_FPS,
        source_fps=Config.SOURCE_FPS,
    )

    if args.max_samples is not None:
        test_dataset.data = test_dataset.data.head(args.max_samples).copy()

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    model = DrivingRiskModel(Config, vocab_size=len(tokenizer)).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    mse_criterion = nn.MSELoss(reduction="mean")
    mae_criterion = nn.L1Loss(reduction="mean")
    mse_scores = []
    mae_scores = []
    references = []
    hypotheses = []

    print(f"Evaluating {len(test_dataset)} samples...")

    for batch in tqdm(test_loader, desc="Evaluate"):
        images = batch["video"].to(device)
        sensors = batch["sensor"].to(device)
        future_targets = batch["future_motion"].to(device)

        pred_caption, pred_motion = generate_caption_and_motion(
            model,
            tokenizer,
            images,
            sensors,
            device,
            max_len=args.max_len,
            beam_size=args.beam_size,
            length_penalty_alpha=args.length_penalty_alpha,
            min_decode_len=args.min_decode_len,
        )

        mse_value = mse_criterion(pred_motion.unsqueeze(0), future_targets).item()
        mse_scores.append(mse_value)

        mae_value = mae_criterion(pred_motion.unsqueeze(0), future_targets).item()
        mae_scores.append(mae_value)

        references.append(tokenizer.decode(batch["caption"][0], skip_special_tokens=True).strip())
        hypotheses.append(pred_caption)

    bleu4_scores = [_sentence_bleu4(ref, hyp) for ref, hyp in zip(references, hypotheses)]
    meteor_scores = [_meteor_score(ref, hyp) for ref, hyp in zip(references, hypotheses)]
    cider, cider_mode = official_cider_score_if_available(references, hypotheses)

    print("\n===== Evaluation Results =====")
    mse_mean = float(np.mean(mse_scores))
    mae_mean = float(np.mean(mae_scores))
    bleu4_mean = float(np.mean(bleu4_scores))
    meteor_mean = float(np.mean(meteor_scores))

    print(f"MSE     : {mse_mean:.6f}")
    print(f"MAE     : {mae_mean:.6f}")
    print(f"BLEU-4  : {bleu4_mean:.6f}")
    print(f"METEOR  : {meteor_mean:.6f}")
    print(f"CIDEr   : {cider:.6f}")
    if cider_mode == "approx":
        print("Note    : pycocoevalcap not found, CIDEr is approximate.")

    # Persist the latest evaluation results for reporting.
    save_path = getattr(args, "save_path", None)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        payload = {
            "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
            "model_path": args.model_path,
            "test_csv": args.test_csv,
            "max_samples": args.max_samples,
            "decode": {
                "max_len": args.max_len,
                "beam_size": args.beam_size if args.beam_size is not None else getattr(Config, "BEAM_SIZE", 5),
                "length_penalty_alpha": args.length_penalty_alpha
                if args.length_penalty_alpha is not None
                else getattr(Config, "LENGTH_PENALTY_ALPHA", 0.7),
                "min_decode_len": args.min_decode_len
                if args.min_decode_len is not None
                else getattr(Config, "MIN_DECODE_LEN", 3),
            },
            "metrics": {
                "mse": mse_mean,
                "mae": mae_mean,
                "bleu4": bleu4_mean,
                "meteor": meteor_mean,
                "cider": float(cider),
                "cider_mode": cider_mode,
            },
        }
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"Saved evaluation summary to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DrivingRisk model on test_data.csv")
    parser.add_argument("--model-path", type=str, default=Config.MODEL_SAVE_PATH)
    parser.add_argument("--test-csv", type=str, default=Config.TEST_CSV)
    parser.add_argument("--max-samples", type=int, default=None, help="Optional quick evaluation limit")
    parser.add_argument("--max-len", type=int, default=30)
    parser.add_argument("--beam-size", type=int, default=None)
    parser.add_argument("--length-penalty-alpha", type=float, default=None)
    parser.add_argument("--min-decode-len", type=int, default=None)
    parser.add_argument(
        "--save-path",
        type=str,
        default=os.path.join("evaluate", "eval_results_latest.json"),
        help="Where to write the latest evaluation results JSON (set to empty to disable).",
    )
    evaluate(parser.parse_args())
