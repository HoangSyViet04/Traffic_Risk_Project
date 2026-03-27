import argparse
import collections
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
from transformers import AutoTokenizer

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


def evaluate(args):
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

    test_dataset = DrivingRiskDataset(
        csv_file=args.test_csv,
        images_root=Config.IMAGES_ROOT,
        telemetry_root=Config.TELEMETRY_ROOT,
        tokenizer=tokenizer,
        transform=transform,
        max_frames=Config.MAX_FRAMES,
        future_steps=Config.FUTURE_STEPS,
    )

    if args.max_samples is not None:
        test_dataset.data = test_dataset.data.head(args.max_samples).copy()

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    model = DrivingRiskModel(Config, vocab_size=len(tokenizer)).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    mse_criterion = nn.MSELoss(reduction="mean")
    mse_scores = []
    references = []
    hypotheses = []

    print(f"Evaluating {len(test_dataset)} samples...")

    for batch in tqdm(test_loader, desc="Evaluate"):
        images = batch["video"].to(device)
        sensors = batch["sensor"].to(device)
        future_targets = batch["future_motion"].to(device)

        pred_caption, pred_motion = generate_caption_and_motion(model, tokenizer, images, sensors, device)

        mse_value = mse_criterion(pred_motion.unsqueeze(0), future_targets).item()  
        mse_scores.append(mse_value)

        references.append(tokenizer.decode(batch["caption"][0], skip_special_tokens=True).strip())
        hypotheses.append(pred_caption)

    bleu4_scores = [_sentence_bleu4(ref, hyp) for ref, hyp in zip(references, hypotheses)]
    meteor_scores = [_meteor_score(ref, hyp) for ref, hyp in zip(references, hypotheses)]
    cider, cider_mode = official_cider_score_if_available(references, hypotheses)

    print("\n===== Evaluation Results =====")
    print(f"MSE     : {float(np.mean(mse_scores)):.6f}")
    print(f"BLEU-4  : {float(np.mean(bleu4_scores)):.6f}")
    print(f"METEOR  : {float(np.mean(meteor_scores)):.6f}")
    print(f"CIDEr   : {cider:.6f}")
    if cider_mode == "approx":
        print("Note    : pycocoevalcap not found, CIDEr is approximate.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DrivingRisk model on test_data.csv")
    parser.add_argument("--model-path", type=str, default=Config.MODEL_SAVE_PATH)
    parser.add_argument("--test-csv", type=str, default=os.path.join(os.path.dirname(Config.TRAIN_CSV), "test_data.csv"))
    parser.add_argument("--max-samples", type=int, default=None, help="Optional quick evaluation limit")
    evaluate(parser.parse_args())
