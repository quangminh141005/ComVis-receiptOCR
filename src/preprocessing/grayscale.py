"""
grayscale.py
------------
Grayscale conversion and upscaling step for receipt preprocessing.

Handles 3 receipt types found in the dataset:
  1. Normal black-ink receipts     -> standard BGR to grayscale
  2. Blue-ink / color receipts     -> weighted channel mix to boost contrast
  3. Already grayscale images      -> returned unchanged
"""

import cv2
import numpy as np


# -- Config defaults (can be overridden via pipeline.py CONFIG) --
DEFAULT_TARGET_MIN_WIDTH      = 0          # 0 = disabled (preserves coordinates)
DEFAULT_UPSCALE_INTERPOLATION = cv2.INTER_CUBIC
BLUE_INK_THRESHOLD            = 10         # min blue-red diff to detect blue ink


def _is_blue_ink(img: np.ndarray) -> bool:
    """
    Detect if receipt uses blue ink by comparing mean Blue vs Red channel.
    Blue-ink receipts lose contrast when converted with standard grayscale
    -- this detects them for special handling.

    Args:
        img (np.ndarray): BGR image.

    Returns:
        bool: True if image appears to have blue-tinted ink.
    """
    b, g, r = cv2.split(img)
    return float(np.mean(b)) > float(np.mean(r)) + BLUE_INK_THRESHOLD


def to_grayscale(img: np.ndarray) -> np.ndarray:
    """
    Convert a BGR image to grayscale using the best method for the receipt type.

    Conversion strategy:
      - Already grayscale (2D)  -> return as-is
      - Blue ink detected       -> weighted R+G blend (drops blue channel)
                                   to maximise text-background contrast
      - Normal black ink        -> standard cv2 BGR->GRAY conversion

    Args:
        img (np.ndarray): Input image (BGR or grayscale).

    Returns:
        np.ndarray: Grayscale image (uint8, same HxW).
    """
    # Already grayscale -- nothing to do
    if len(img.shape) == 2:
        return img

    # Blue-ink receipt -> boost contrast by dropping the blue channel
    if _is_blue_ink(img):
        print("    [grayscale] blue-ink receipt detected -- using R+G blend")
        b, g, r = cv2.split(img)
        # Weight red and green equally; ignore blue channel
        gray = cv2.addWeighted(r, 0.5, g, 0.5, 0)
        return gray

    # Standard black-ink receipt
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def upscale_if_needed(
    img: np.ndarray,
    min_width: int = DEFAULT_TARGET_MIN_WIDTH,
    interpolation: int = DEFAULT_UPSCALE_INTERPOLATION,
) -> np.ndarray:
    """
    Upscale the image proportionally if its width is below min_width.

    WARNING: Upscaling invalidates the bounding-box coordinates in
    the paired .txt annotation files. Keep min_width = 0 (disabled)
    whenever ground-truth coordinates must stay aligned with the image.

    Args:
        img          (np.ndarray): Input image.
        min_width    (int)       : Minimum width in pixels.
                                   Set to 0 to disable upscaling entirely.
        interpolation            : OpenCV interpolation flag.

    Returns:
        np.ndarray: Upscaled (or original) image.
    """
    # Disabled
    if min_width == 0:
        return img

    h, w = img.shape[:2]
    if w < min_width:
        scale = min_width / w
        new_w = int(w * scale)
        new_h = int(h * scale)
        img   = cv2.resize(img, (new_w, new_h), interpolation=interpolation)
        print(f"    [upscale] {w}x{h} -> {new_w}x{new_h}  (scale={scale:.2f})")
    return img