"""
evaluate.py
-----------
Evaluates preprocessing quality by comparing Tesseract OCR output
on original vs preprocessed images using CER and WER metrics.

Usage:
    python evaluate.py --original "data/0325updated.task1train(626p)" \
                       --preprocessed "data/preprocessed" \
                       --gt_dir "data/0325updated.task1train(626p)" \
                       --max_samples 20
"""

import os
import re
import argparse
import difflib
from pathlib import Path

import cv2
import pytesseract
import numpy as np

# ── Point to your Tesseract install (Windows) ──────────────────────────────
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Tesseract config: treat image as a single block of text
TESS_CONFIG = "--oem 1 --psm 6"


# ─────────────────────────────────────────────
#  TEXT NORMALIZATION
# ─────────────────────────────────────────────

def normalize(text: str) -> str:
    """Lowercase, collapse whitespace, strip punctuation for fair comparison."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ─────────────────────────────────────────────
#  METRICS
# ─────────────────────────────────────────────

def character_error_rate(pred: str, gt: str) -> float:
    """
    CER = edit distance at character level / length of ground truth.
    Lower is better. 0.0 = perfect, 1.0 = completely wrong.
    """
    if len(gt) == 0:
        return 0.0 if len(pred) == 0 else 1.0
    matcher = difflib.SequenceMatcher(None, pred, gt)
    edit_dist = len(gt) + len(pred) - 2 * sum(
        triple.size for triple in matcher.get_matching_blocks()
    )
    return edit_dist / len(gt)


def word_error_rate(pred: str, gt: str) -> float:
    """
    WER = edit distance at word level / number of words in ground truth.
    Lower is better.
    """
    pred_words = pred.split()
    gt_words   = gt.split()
    if len(gt_words) == 0:
        return 0.0 if len(pred_words) == 0 else 1.0
    matcher = difflib.SequenceMatcher(None, pred_words, gt_words)
    edit_dist = len(gt_words) + len(pred_words) - 2 * sum(
        triple.size for triple in matcher.get_matching_blocks()
    )
    return edit_dist / len(gt_words)


# ─────────────────────────────────────────────
#  GROUND TRUTH LOADER
# ─────────────────────────────────────────────

def load_ground_truth(txt_path: str) -> str:
    """
    Load text from annotation .txt file.
    Format per line: x1,y1,x2,y2,...,TEXT
    Extracts only the TEXT portion from each line.
    """
    lines = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Split on comma, last element is the text label
            parts = line.split(",")
            if len(parts) >= 9:
                text = ",".join(parts[8:])  # handle commas in text
                lines.append(text)
    return " ".join(lines)


# ─────────────────────────────────────────────
#  OCR RUNNER
# ─────────────────────────────────────────────

def run_tesseract(image_path: str) -> str:
    """Run Tesseract OCR on an image and return extracted text."""
    img = cv2.imread(image_path)
    if img is None:
        return ""
    return pytesseract.image_to_string(img, config=TESS_CONFIG)


# ─────────────────────────────────────────────
#  MAIN EVALUATION
# ─────────────────────────────────────────────

def evaluate(original_dir: str, preprocessed_dir: str,
             gt_dir: str, max_samples: int = 20):

    original_path     = Path(original_dir)
    preprocessed_path = Path(preprocessed_dir)
    gt_path           = Path(gt_dir)

    # Gather image pairs
    image_files = sorted(list(original_path.rglob("*.jpg")) +
                         list(original_path.rglob("*.png")))[:max_samples]

    if not image_files:
        print("No images found in original directory.")
        return

    results = []

    print(f"\n{'='*70}")
    print(f"  Evaluating {len(image_files)} images with Tesseract OCR")
    print(f"{'='*70}\n")

    for img_file in image_files:
        rel          = img_file.relative_to(original_path)
        pre_file     = preprocessed_path / rel
        gt_file      = gt_path / img_file.relative_to(original_path)
        gt_file      = gt_file.with_suffix(".txt")

        if not pre_file.exists():
            print(f"  [SKIP] No preprocessed version found: {rel}")
            continue
        if not gt_file.exists():
            print(f"  [SKIP] No ground truth .txt found: {rel}")
            continue

        # Load ground truth
        gt_text   = normalize(load_ground_truth(str(gt_file)))

        # Run OCR on both
        orig_text = normalize(run_tesseract(str(img_file)))
        pre_text  = normalize(run_tesseract(str(pre_file)))

        # Compute metrics
        orig_cer = character_error_rate(orig_text, gt_text)
        pre_cer  = character_error_rate(pre_text,  gt_text)
        orig_wer = word_error_rate(orig_text, gt_text)
        pre_wer  = word_error_rate(pre_text,  gt_text)

        improvement_cer = orig_cer - pre_cer   # positive = improved
        improvement_wer = orig_wer - pre_wer

        results.append({
            "file":        str(rel),
            "orig_cer":    orig_cer,
            "pre_cer":     pre_cer,
            "orig_wer":    orig_wer,
            "pre_wer":     pre_wer,
            "delta_cer":   improvement_cer,
            "delta_wer":   improvement_wer,
        })

        status = "✅ IMPROVED" if improvement_cer > 0 else \
                 "➖ SAME    " if improvement_cer == 0 else "❌ WORSE   "

        print(f"  {status}  {rel.name}")
        print(f"           CER: {orig_cer:.3f} → {pre_cer:.3f}  "
              f"({'↓' if improvement_cer>0 else '↑'}{abs(improvement_cer):.3f})")
        print(f"           WER: {orig_wer:.3f} → {pre_wer:.3f}  "
              f"({'↓' if improvement_wer>0 else '↑'}{abs(improvement_wer):.3f})\n")

    # ── Summary ──────────────────────────────────────────────────────────
    if not results:
        print("No results to summarize.")
        return

    avg_orig_cer = np.mean([r["orig_cer"] for r in results])
    avg_pre_cer  = np.mean([r["pre_cer"]  for r in results])
    avg_orig_wer = np.mean([r["orig_wer"] for r in results])
    avg_pre_wer  = np.mean([r["pre_wer"]  for r in results])

    improved = sum(1 for r in results if r["delta_cer"] > 0)
    worse    = sum(1 for r in results if r["delta_cer"] < 0)
    same     = sum(1 for r in results if r["delta_cer"] == 0)

    print(f"{'='*70}")
    print(f"  SUMMARY  ({len(results)} images evaluated)")
    print(f"{'='*70}")
    print(f"  {'Metric':<10} {'Original':>10} {'Preprocessed':>14} {'Change':>10}")
    print(f"  {'-'*46}")
    print(f"  {'Avg CER':<10} {avg_orig_cer:>10.3f} {avg_pre_cer:>14.3f} "
          f"{avg_pre_cer - avg_orig_cer:>+10.3f}")
    print(f"  {'Avg WER':<10} {avg_orig_wer:>10.3f} {avg_pre_wer:>14.3f} "
          f"{avg_pre_wer - avg_orig_wer:>+10.3f}")
    print(f"\n  Images improved : {improved}")
    print(f"  Images worse    : {worse}")
    print(f"  Images same     : {same}")
    print(f"\n  {'✅ Preprocessing HELPS!' if avg_pre_cer < avg_orig_cer else '❌ Preprocessing HURTS — tune your parameters'}")
    print(f"{'='*70}\n")


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate preprocessing with Tesseract OCR")
    parser.add_argument("--original",     default="data/0325updated.task1train(626p)")
    parser.add_argument("--preprocessed", default="data/preprocessed")
    parser.add_argument("--gt_dir",       default="data/0325updated.task1train(626p)")
    parser.add_argument("--max_samples",  type=int, default=20,
                        help="Number of images to evaluate (default: 20)")
    args = parser.parse_args()

    evaluate(args.original, args.preprocessed, args.gt_dir, args.max_samples)