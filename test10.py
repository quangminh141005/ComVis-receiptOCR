"""
test10.py
---------
Preprocesses and evaluates exactly 10 specific images.
Saves results to results.json for visualize_comparison.py to read.

Usage:
    python test10.py
"""

import os
import re
import json
import difflib
from pathlib import Path

import cv2
import numpy as np
import pytesseract

# -- Tesseract path ----------------------------------------------------------
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
TESS_CONFIG = "--oem 1 --psm 6"

# -- The 10 images to test ---------------------------------------------------
TEST_IMAGES = [
    "X51007843145.jpg",
    "X51007579726.jpg",
    "X51008030560.jpg",
    "X51008122920.jpg",
    "X51008164992.jpg",
    "X51008063849.jpg",
    "X51007339135.jpg",
    "X51005806698.jpg",
    "X51005745187.jpg",
    "X51005719862.jpg",
]

# -- Paths -------------------------------------------------------------------
ORIGINAL_DIR     = "data/0325updated.task1train(626p)"
PREPROCESSED_DIR = "data/preprocessed_10test"
RESULTS_JSON     = "results.json"

# -- Imports (only what is actually used) ------------------------------------
from src.preprocessing.grayscale    import to_grayscale
from src.preprocessing.denoising    import denoise
from src.preprocessing.contrast     import enhance_contrast
from src.preprocessing.thresholding import binarize
from src.preprocessing.morphology   import morphological_cleanup

# ----------------------------------------------------------------------------
#  CONFIG
#  - Toggle each step with its _enabled flag
#  - upscale / deskew / crop are permanently OFF (they break coordinates)
# ----------------------------------------------------------------------------
CONFIG = {
    # Grayscale — always runs, no toggle needed
    # (auto-detects blue-ink receipts)

    # Denoise
    "denoise_enabled":     True,
    "denoise_h":           6,        # 1-20: higher = more smoothing

    # Contrast (CLAHE) — enable for faded/low-contrast receipts
    "clahe_enabled":       False,
    "clahe_clip_limit":    1.5,      # 0.5-3.0
    "clahe_tile_grid":     (8, 8),

    # Binarize — always runs
    "thresh_block_size":   53,       # must be odd; larger handles uneven lighting
    "thresh_c":            10,       # higher removes more background noise

    # Morphology
    "morph_enabled":       True,
    "morph_kernel_size":   (1, 1),   # increase to (2,2) for broken strokes
}


# ----------------------------------------------------------------------------
#  PREPROCESSING
# ----------------------------------------------------------------------------
def preprocess_image(image_path: str, output_path: str) -> bool:
    img = cv2.imread(image_path)
    if img is None:
        print(f"  [ERROR] Cannot read: {image_path}")
        return False

    # Step 1 — Grayscale (always on; auto blue-ink detection)
    gray = to_grayscale(img)

    # Step 2 — Denoise
    if CONFIG["denoise_enabled"]:
        gray = denoise(gray, h=CONFIG["denoise_h"])

    # Step 3 — Contrast enhancement (CLAHE)
    if CONFIG["clahe_enabled"]:
        gray = enhance_contrast(
            gray,
            clip_limit=CONFIG["clahe_clip_limit"],
            tile_grid=CONFIG["clahe_tile_grid"],
        )

    # Step 4 — Binarize (always runs)
    binary = binarize(
        gray,
        block_size=CONFIG["thresh_block_size"],
        c=CONFIG["thresh_c"],
    )

    # Step 5 — Morphological cleanup
    if CONFIG["morph_enabled"]:
        binary = morphological_cleanup(
            binary,
            kernel_size=CONFIG["morph_kernel_size"],
        )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cv2.imwrite(output_path, binary)
    return True


# ----------------------------------------------------------------------------
#  EVALUATION HELPERS
# ----------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------
#  MAIN
# ----------------------------------------------------------------------------
def main():
    original_path     = Path(ORIGINAL_DIR)
    preprocessed_path = Path(PREPROCESSED_DIR)
    preprocessed_path.mkdir(parents=True, exist_ok=True)

    # Print active steps
    print("\n" + "="*70)
    print("  ACTIVE PIPELINE")
    print("="*70)
    steps = ["1. Grayscale (always on — auto blue-ink detection)"]
    n = 2
    if CONFIG["denoise_enabled"]:
        steps.append(f"{n}. Denoise  (h={CONFIG['denoise_h']})")
        n += 1
    if CONFIG["clahe_enabled"]:
        steps.append(f"{n}. CLAHE  (clip={CONFIG['clahe_clip_limit']})")
        n += 1
    steps.append(f"{n}. Binarize  (block={CONFIG['thresh_block_size']}, c={CONFIG['thresh_c']})")
    n += 1
    if CONFIG["morph_enabled"]:
        steps.append(f"{n}. Morphology  (kernel={CONFIG['morph_kernel_size']})")
    for s in steps:
        print(f"  {s}")

    # -- STEP 1: Preprocess --------------------------------------------------
    print("\n" + "="*70)
    print("  STEP 1 — PREPROCESSING")
    print("="*70)

    success = []
    for fname in TEST_IMAGES:
        matches = list(original_path.rglob(fname))
        if not matches:
            print(f"  [NOT FOUND] {fname}")
            continue
        src = matches[0]
        dst = preprocessed_path / fname
        print(f"  {fname} ...", end=" ", flush=True)
        if preprocess_image(str(src), str(dst)):
            print("Done")
            success.append(fname)
        else:
            print("FAILED")

    print(f"\n  {len(success)}/{len(TEST_IMAGES)} images preprocessed successfully.")

    # -- STEP 2: Evaluate ----------------------------------------------------
    print("\n" + "="*70)
    print("  STEP 2 — EVALUATION")
    print("="*70 + "\n")

    results = {}

    for fname in success:
        matches  = list(original_path.rglob(fname))
        orig_img = str(matches[0])
        pre_img  = str(preprocessed_path / fname)
        txt_file = Path(orig_img).with_suffix(".txt")

        if not txt_file.exists():
            print(f"  [SKIP] No .txt found for {fname}")
            continue

        gt_text   = normalize(load_ground_truth(str(txt_file)))
        orig_text = normalize(run_tesseract(orig_img))
        pre_text  = normalize(run_tesseract(pre_img))

        orig_cer  = character_error_rate(orig_text, gt_text)
        pre_cer   = character_error_rate(pre_text,  gt_text)
        orig_wer  = word_error_rate(orig_text, gt_text)
        pre_wer   = word_error_rate(pre_text,  gt_text)
        delta_cer = orig_cer - pre_cer
        delta_wer = orig_wer - pre_wer

        status = "IMPROVED" if delta_cer >  0.01 else \
                 "WORSE"    if delta_cer < -0.01 else "SAME"

        results[fname] = {
            "orig_cer": round(orig_cer, 3),
            "pre_cer":  round(pre_cer,  3),
            "orig_wer": round(orig_wer, 3),
            "pre_wer":  round(pre_wer,  3),
            "status":   status,
        }

        icon = "IMPROVED" if status == "IMPROVED" else \
               "WORSE   " if status == "WORSE"    else "SAME    "
        cer_dir = "down" if delta_cer > 0 else "up"
        wer_dir = "down" if delta_wer > 0 else "up"
        print(f"  {icon}  {fname}")
        print(f"    CER: {orig_cer:.3f} -> {pre_cer:.3f}  ({cer_dir} {abs(delta_cer):.3f})")
        print(f"    WER: {orig_wer:.3f} -> {pre_wer:.3f}  ({wer_dir} {abs(delta_wer):.3f})\n")

    # -- Save JSON -----------------------------------------------------------
    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved -> {RESULTS_JSON}\n")

    # -- Summary -------------------------------------------------------------
    if not results:
        return

    vals         = list(results.values())
    avg_orig_cer = np.mean([r["orig_cer"] for r in vals])
    avg_pre_cer  = np.mean([r["pre_cer"]  for r in vals])
    avg_orig_wer = np.mean([r["orig_wer"] for r in vals])
    avg_pre_wer  = np.mean([r["pre_wer"]  for r in vals])
    improved     = sum(1 for r in vals if r["status"] == "IMPROVED")
    worse        = sum(1 for r in vals if r["status"] == "WORSE")
    same         = sum(1 for r in vals if r["status"] == "SAME")

    print("="*70)
    print(f"  SUMMARY  ({len(vals)} images evaluated)")
    print("="*70)
    print(f"  {'Metric':<10} {'Original':>10} {'Preprocessed':>14} {'Change':>10}")
    print(f"  {'-'*46}")
    print(f"  {'Avg CER':<10} {avg_orig_cer:>10.3f} {avg_pre_cer:>14.3f} "
          f"{avg_pre_cer - avg_orig_cer:>+10.3f}")
    print(f"  {'Avg WER':<10} {avg_orig_wer:>10.3f} {avg_pre_wer:>14.3f} "
          f"{avg_pre_wer - avg_orig_wer:>+10.3f}")
    print(f"\n  Improved : {improved}  |  Worse : {worse}  |  Same : {same}")
    verdict = "Preprocessing HELPS!" if avg_pre_cer < avg_orig_cer \
              else "Preprocessing HURTS -- tune your parameters"
    print(f"\n  {verdict}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()