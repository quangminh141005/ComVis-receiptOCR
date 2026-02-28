"""
pipeline.py
-----------
Orchestrates all preprocessing steps for receipt OCR.
Imports each step from its dedicated module in src/preprocessing/.

Usage (from project root):
    # Process entire dataset
    python pipeline.py --mode batch

    # Process a single image
    python pipeline.py --mode single --input data/sample.jpg --output out.jpg

    # Visualise each step side-by-side (debug / parameter tuning)
    python pipeline.py --mode visualise --input data/sample.jpg
"""

import os
import shutil
import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# ── Import individual preprocessing steps ──────────────────────────────────
from src.preprocessing.grayscale    import to_grayscale, upscale_if_needed
from src.preprocessing.deskewing    import deskew_image
from src.preprocessing.denoising    import denoise
from src.preprocessing.contrast     import enhance_contrast
from src.preprocessing.thresholding import binarize
from src.preprocessing.morphology   import morphological_cleanup
from src.preprocessing.cropping     import crop_to_content


# ─────────────────────────────────────────────
#  CONFIGURATION  (tune parameters here)
# ─────────────────────────────────────────────
CONFIG = {
    # Paths
    "input_dir":      "data/0325updated.task1train(626p)",
    "output_dir":     "data/preprocessed",
    "img_extensions": ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif"],

    # Step: upscaling
    "target_min_width": 0,

    # Step: denoising
    "denoise_h": 5,

    # Step: contrast (CLAHE)
    "clahe_clip_limit": 0.0,
    "clahe_tile_grid":  (8, 8),

    # Step: binarization
    "thresh_block_size": 41,   # must be odd
    "thresh_c":          3,

    # Step: morphological cleanup
    "morph_kernel_size": (1, 1),

    # Step: deskew
    "deskew_enabled": False ,
    "deskew_bg_color": 255,

    # Step: border crop
    "border_crop_enabled": False,
    "border_padding":      10,
}


# ─────────────────────────────────────────────
#  SINGLE-IMAGE PIPELINE
# ─────────────────────────────────────────────
def preprocess(image_path: str, output_path: str) -> None:
    """Run the full preprocessing pipeline on one image and save the result."""

    img = cv2.imread(image_path)
    if img is None:
        print(f"  [WARN] Could not read: {image_path}")
        return

    # 1. Upscale if too small
    img = upscale_if_needed(img, min_width=CONFIG["target_min_width"])

    # 2. Grayscale
    gray = to_grayscale(img)

    # 3. Deskew
    if CONFIG["deskew_enabled"]:
        gray = deskew_image(gray, bg_color=CONFIG["deskew_bg_color"])

    # 4. Denoise
    denoised = denoise(gray, h=CONFIG["denoise_h"])

    # 5. Contrast enhancement (CLAHE)
    enhanced = denoised #enhance_contrast(
    #    denoised,
    #    clip_limit=CONFIG["clahe_clip_limit"],
    #    tile_grid=CONFIG["clahe_tile_grid"],
    #)
    
    # 6. Binarize
    binary = binarize(
        enhanced,
        block_size=CONFIG["thresh_block_size"],
        c=CONFIG["thresh_c"],
    )

    # 7. Morphological cleanup
    cleaned = morphological_cleanup(binary, kernel_size=CONFIG["morph_kernel_size"])

    # 8. Crop to content
    if CONFIG["border_crop_enabled"]:
        cleaned = crop_to_content(cleaned, cleaned, padding=CONFIG["border_padding"])

    # 9. Save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cv2.imwrite(output_path, cleaned)


# ─────────────────────────────────────────────
#  BATCH PROCESSING
# ─────────────────────────────────────────────
def batch_preprocess(input_dir: str, output_dir: str) -> None:
    """
    Walk input_dir, preprocess every receipt image, mirror the folder
    structure under output_dir. Paired .txt annotation files are copied as-is.
    """
    input_path  = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    image_files = []
    for ext in CONFIG["img_extensions"]:
        image_files.extend(input_path.rglob(ext))
    image_files = sorted(image_files)

    if not image_files:
        print(f"No images found in: {input_dir}")
        return

    print(f"\nFound {len(image_files)} images — starting preprocessing...\n")

    for img_file in tqdm(image_files, desc="Preprocessing", unit="img"):
        rel      = img_file.relative_to(input_path)
        out_file = output_path / rel

        print(f"  Processing: {rel}")
        preprocess(str(img_file), str(out_file))

        # Copy paired annotation .txt file if present
        txt_file = img_file.with_suffix(".txt")
        if txt_file.exists():
            out_txt = output_path / txt_file.relative_to(input_path)
            out_txt.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(txt_file, out_txt)

    print(f"\n✅  Done! Preprocessed images saved to: {output_dir}\n")


# ─────────────────────────────────────────────
#  VISUALISE PIPELINE (DEBUG)
# ─────────────────────────────────────────────
def visualise_pipeline(image_path: str) -> None:
    """Show each preprocessing step side-by-side for parameter tuning."""

    img      = cv2.imread(image_path)
    img      = upscale_if_needed(img, CONFIG["target_min_width"])
    gray     = to_grayscale(img)
    deskewed = deskew_image(gray) if CONFIG["deskew_enabled"] else gray
    denoised = denoise(deskewed, h=CONFIG["denoise_h"])
    enhanced = enhance_contrast(denoised,
                                clip_limit=CONFIG["clahe_clip_limit"],
                                tile_grid=CONFIG["clahe_tile_grid"])
    binary   = binarize(enhanced,
                        block_size=CONFIG["thresh_block_size"],
                        c=CONFIG["thresh_c"])
    cleaned  = morphological_cleanup(binary, CONFIG["morph_kernel_size"])
    cropped  = crop_to_content(cleaned, cleaned, CONFIG["border_padding"]) \
               if CONFIG["border_crop_enabled"] else cleaned

    steps = [
        ("1. Grayscale",       gray),
        ("2. Deskewed",        deskewed),
        ("3. Denoised",        denoised),
        ("4. CLAHE Enhanced",  enhanced),
        ("5. Binarized",       binary),
        ("6. Morph Cleanup",   cleaned),
        ("7. Cropped",         cropped),
    ]

    target_h = 600
    panels   = []
    for title, step_img in steps:
        h, w   = step_img.shape[:2]
        scale  = target_h / h
        resized = cv2.resize(step_img, (int(w * scale), target_h))
        labeled = cv2.copyMakeBorder(resized, 30, 0, 0, 0,
                                     cv2.BORDER_CONSTANT, value=200)
        cv2.putText(labeled, title, (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 2)
        panels.append(labeled)

    mid  = len(panels) // 2 + len(panels) % 2
    row1 = np.hstack(panels[:mid])
    row2 = np.hstack(panels[mid:])
    if row2.shape[1] < row1.shape[1]:
        pad  = row1.shape[1] - row2.shape[1]
        row2 = cv2.copyMakeBorder(row2, 0, 0, 0, pad,
                                  cv2.BORDER_CONSTANT, value=255)
    grid = np.vstack([row1, row2])

    cv2.imshow("Preprocessing Pipeline — each step", grid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Receipt OCR Preprocessing Pipeline")
    parser.add_argument(
        "--mode", choices=["batch", "single", "visualise"],
        default="batch",
        help="batch: process entire dataset | "
             "single: preprocess one image | "
             "visualise: show step-by-step for one image",
    )
    parser.add_argument("--input",  default=CONFIG["input_dir"])
    parser.add_argument("--output", default=CONFIG["output_dir"])
    args = parser.parse_args()

    if args.mode == "batch":
        batch_preprocess(args.input, args.output)

    elif args.mode == "single":
        out = args.output if args.output != CONFIG["output_dir"] \
              else "preprocessed_output.jpg"
        preprocess(args.input, out)
        print(f"Saved → {out}")

    elif args.mode == "visualise":
        visualise_pipeline(args.input)