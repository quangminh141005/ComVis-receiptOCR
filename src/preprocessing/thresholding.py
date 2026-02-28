"""
thresholding.py
---------------
Binarization (adaptive thresholding) step for receipt preprocessing.
"""

import cv2
import numpy as np


# ── Config defaults ──
DEFAULT_THRESH_BLOCK_SIZE = 31   # must be odd
DEFAULT_THRESH_C          = 10


def binarize(
    gray: np.ndarray,
    block_size: int = DEFAULT_THRESH_BLOCK_SIZE,
    c: int          = DEFAULT_THRESH_C,
) -> np.ndarray:
    """
    Apply Adaptive Gaussian Thresholding to produce a clean binary image.

    Adaptive (local) thresholding is preferred over global Otsu for receipts
    because thermal-paper images often have uneven lighting, shadows, or
    faded regions that a single global threshold handles poorly.

    Args:
        gray       (np.ndarray): Grayscale input image.
        block_size (int)       : Size of the local neighbourhood window
                                 (must be an odd number ≥ 3).
        c          (int)       : Constant subtracted from the local mean.
                                 Higher values → thinner strokes.

    Returns:
        np.ndarray: Binary (black-and-white) image.
    """
    assert block_size % 2 == 1 and block_size >= 3, \
        "block_size must be an odd integer ≥ 3"

    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        c,
    )