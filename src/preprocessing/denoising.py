"""
denoising.py
------------
Noise removal step for receipt preprocessing.
"""

import cv2
import numpy as np


# ── Config defaults ──
DEFAULT_DENOISE_H = 10   # filter strength; higher = more smoothing


def denoise(
    gray: np.ndarray,
    h: int = DEFAULT_DENOISE_H,
) -> np.ndarray:
    """
    Remove salt-and-pepper / sensor noise from a grayscale receipt image
    using Non-Local Means Denoising.

    NLM is preferred over simple blurs (Gaussian, median) because it
    suppresses noise while preserving fine text strokes — a critical
    property when characters are small (some bounding boxes in receipts
    are only ~15 px tall).

    Args:
        gray (np.ndarray): Grayscale input image.
        h    (int)       : Filter strength.
                           • 5–10  → light noise, well-exposed images
                           • 10–20 → heavier noise or low-resolution scans
                           Values above 20 may start to blur fine strokes.

    Returns:
        np.ndarray: Denoised grayscale image.
    """
    return cv2.fastNlMeansDenoising(gray, h=h)