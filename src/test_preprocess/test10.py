"""
test10.py
---------
Tests multiple preprocessing variants (P0-P6) on 10 images,
evaluates each with Tesseract OCR, and saves results to results.json.

Variants (inspired by the PaddleOCR approach):
    P0 : no preprocessing (original)
    P1 : grayscale only
    P2 : grayscale + bilateral denoise
    P3 : grayscale + bilateral + CLAHE
    P4 : grayscale + bilateral + binarize
    P5 : grayscale + bilateral + CLAHE + binarize
    P6 : grayscale + bilateral + CLAHE + sharpen + binarize

Usage:
    python test10.py
"""

import os
import re
import json
import difflib
from pathlib import Path
from typing import Callable, Dict

import cv2
import numpy as np
import pytesseract

# -- Tesseract path ----------------------------------------------------------
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
TESS_CONFIG = "--oem 1 --psm 6"

# -- The 10 images to test ---------------------------------------------------
TEST_IMAGES = [
    "X00016469612.jpg",
    "X51005255805.jpg",
    "X51005268472.jpg",
    "X51008164969.jpg",
    "X51008142063.jpg",
    "X51008099083.jpg",
    "X51008099081.jpg",
    "X51007262315.jpg",
    "X51007231370.jpg",
    "X51007228448.jpg",
]

# -- Paths -------------------------------------------------------------------
ORIGINAL_DIR     = "data/0325updated.task1train(626p)"
PREPROCESSED_DIR = "data/preprocessed_variants"
RESULTS_JSON     = "results.json"

# -- Import all functions from single preprocessing file ---------------------
# Change these imports in test10.py
from src.preprocessing.preprocessing import (
    to_grayscale,
    denoise_nlm,
    denoise_bilateral,
    enhance_contrast,
    sharpen,
    binarize,
    morphological_cleanup,
)

# -- Threshold params (shared across variants that binarize) -----------------
BLOCK_SIZE = 41
C          = 10


# ============================================================================
#  VARIANTS  — each is a function: BGR image -> processed image
# ============================================================================

def P0_original(img: np.ndarray) -> np.ndarray:
    """No preprocessing — pass through as-is."""
    return img


def P1_gray(img: np.ndarray) -> np.ndarray:
    """Grayscale only."""
    return to_grayscale(img)


def P2_gray_bilateral(img: np.ndarray) -> np.ndarray:
    """Grayscale + bilateral denoise."""
    gray = to_grayscale(img)
    gray = denoise_bilateral(gray)
    return gray


def P3_gray_bilateral_clahe(img: np.ndarray) -> np.ndarray:
    """Grayscale + bilateral + CLAHE contrast."""
    gray = to_grayscale(img)
    gray = denoise_bilateral(gray)
    gray = enhance_contrast(gray)
    return gray


def P4_gray_bilateral_binarize(img: np.ndarray) -> np.ndarray:
    """Grayscale + bilateral + binarize."""
    gray   = to_grayscale(img)
    gray   = denoise_bilateral(gray)
    binary = binarize(gray, block_size=BLOCK_SIZE, c=C)
    return binary



# Map name -> function
VARIANTS: Dict[str, Callable] = {
    "P0_original":                       P0_original,
    "P1_gray":                           P1_gray,
    "P2_gray_bilateral":                 P2_gray_bilateral,
    "P3_gray_bilateral_clahe":           P3_gray_bilateral_clahe,
    "P4_gray_bilateral_binarize":        P4_gray_bilateral_binarize,
    
}


# ============================================================================
#  EVALUATION HELPERS
# ============================================================================

def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def character_error_rate(pred: str, gt: str) -> float:
    if len(gt) == 0:
        return 0.0 if len(pred) == 0 else 1.0
    matcher   = difflib.SequenceMatcher(None, pred, gt)
    edit_dist = len(gt) + len(pred) - 2 * sum(
        t.size for t in matcher.get_matching_blocks())
    return edit_dist / len(gt)


def word_error_rate(pred: str, gt: str) -> float:
    pw = pred.split()
    gw = gt.split()
    if len(gw) == 0:
        return 0.0 if len(pw) == 0 else 1.0
    matcher   = difflib.SequenceMatcher(None, pw, gw)
    edit_dist = len(gw) + len(pw) - 2 * sum(
        t.size for t in matcher.get_matching_blocks())
    return edit_dist / len(gw)


def load_ground_truth(txt_path: str) -> str:
    lines = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) >= 9:
                lines.append(",".join(parts[8:]))
    return " ".join(lines)


def run_tesseract(image_path: str) -> str:
    img = cv2.imread(image_path)
    if img is None:
        return ""
    return pytesseract.image_to_string(img, config=TESS_CONFIG)


def save_variant_image(img: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)


# ============================================================================
#  MAIN
# ============================================================================

def main():
    original_path = Path(ORIGINAL_DIR)

    print("\n" + "="*70)
    print(f"  TESTING {len(VARIANTS)} VARIANTS ON {len(TEST_IMAGES)} IMAGES")
    print("="*70)
    for vname in VARIANTS:
        print(f"  - {vname}")

    # ── Collect ground-truth image paths ────────────────────────────────────
    image_paths = {}
    for fname in TEST_IMAGES:
        matches = list(original_path.rglob(fname))
        if matches:
            image_paths[fname] = matches[0]
        else:
            print(f"  [NOT FOUND] {fname}")

    # ── Run all variants on all images ───────────────────────────────────────
    print("\n" + "="*70)
    print("  STEP 1 — PREPROCESSING ALL VARIANTS")
    print("="*70)

    # results[variant][image] = {orig_cer, pre_cer, orig_wer, pre_wer, status}
    all_results: Dict[str, Dict] = {v: {} for v in VARIANTS}
    # cache original OCR + GT so we don't re-run for every variant
    orig_ocr_cache = {}
    gt_cache       = {}

    for fname, img_path in image_paths.items():
        txt_file = img_path.with_suffix(".txt")
        if not txt_file.exists():
            print(f"  [SKIP] No .txt for {fname}")
            continue

        gt_cache[fname]       = normalize(load_ground_truth(str(txt_file)))
        orig_ocr_cache[fname] = normalize(run_tesseract(str(img_path)))

        img_bgr = cv2.imread(str(img_path))

        for vname, vfn in VARIANTS.items():
            out_path = Path(PREPROCESSED_DIR) / vname / fname
            processed = vfn(img_bgr)
            save_variant_image(processed, out_path)

        print(f"  {fname} ... all variants done")

    # ── Evaluate ─────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("  STEP 2 — EVALUATION PER VARIANT")
    print("="*70)

    variant_summary = {}   # vname -> {avg_cer, avg_wer, improved, worse, same}

    for vname, vfn in VARIANTS.items():
        print(f"\n  [{vname}]")
        v_results = {}
        cers, wers = [], []

        for fname in image_paths:
            if fname not in gt_cache:
                continue

            pre_path = Path(PREPROCESSED_DIR) / vname / fname
            if not pre_path.exists():
                continue

            gt_text   = gt_cache[fname]
            orig_text = orig_ocr_cache[fname]
            pre_text  = normalize(run_tesseract(str(pre_path)))

            orig_cer  = character_error_rate(orig_text, gt_text)
            pre_cer   = character_error_rate(pre_text,  gt_text)
            orig_wer  = word_error_rate(orig_text, gt_text)
            pre_wer   = word_error_rate(pre_text,  gt_text)
            delta_cer = orig_cer - pre_cer
            delta_wer = orig_wer - pre_wer

            status = "IMPROVED" if delta_cer >  0.01 else \
                     "WORSE"    if delta_cer < -0.01 else "SAME"

            v_results[fname] = {
                "orig_cer": round(orig_cer, 3),
                "pre_cer":  round(pre_cer,  3),
                "orig_wer": round(orig_wer, 3),
                "pre_wer":  round(pre_wer,  3),
                "status":   status,
            }
            cers.append(pre_cer)
            wers.append(pre_wer)

            icon    = "IMPROVED" if status == "IMPROVED" else \
                      "WORSE   " if status == "WORSE"    else "SAME    "
            cer_dir = "down" if delta_cer > 0 else "up"
            wer_dir = "down" if delta_wer > 0 else "up"
            print(f"    {icon}  {fname}")
            print(f"      CER: {orig_cer:.3f} -> {pre_cer:.3f}  ({cer_dir} {abs(delta_cer):.3f})")
            print(f"      WER: {orig_wer:.3f} -> {pre_wer:.3f}  ({wer_dir} {abs(delta_wer):.3f})")

        all_results[vname] = v_results

        if cers:
            variant_summary[vname] = {
                "avg_cer":  round(float(np.mean(cers)), 3),
                "avg_wer":  round(float(np.mean(wers)), 3),
                "improved": sum(1 for r in v_results.values() if r["status"] == "IMPROVED"),
                "worse":    sum(1 for r in v_results.values() if r["status"] == "WORSE"),
                "same":     sum(1 for r in v_results.values() if r["status"] == "SAME"),
            }

    # ── Save JSON ────────────────────────────────────────────────────────────
    output = {
        "per_variant": all_results,
        "summary":     variant_summary,
    }
    with open(RESULTS_JSON, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved -> {RESULTS_JSON}")

    # ── Final leaderboard ────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("  LEADERBOARD  (ranked by Avg CER — lower is better)")
    print("="*70)
    print(f"  {'Variant':<45} {'Avg CER':>8} {'Avg WER':>8} {'Improved':>9} {'Worse':>6}")
    print(f"  {'-'*78}")

    ranked = sorted(variant_summary.items(), key=lambda x: x[1]["avg_cer"])
    for i, (vname, s) in enumerate(ranked):
        medal = "BEST  " if i == 0 else "      "
        print(f"  {medal}{vname:<39} {s['avg_cer']:>8.3f} {s['avg_wer']:>8.3f} "
              f"{s['improved']:>9} {s['worse']:>6}")

    best = ranked[0][0]
    print(f"\n  Best variant: {best}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()