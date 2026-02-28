"""
contrast.py
-----------
Contrast enhancement (CLAHE) step for receipt preprocessing.
"""

import cv2
import numpy as np


# ── Config defaults ──
DEFAULT_CLIP_LIMIT  = 2.0
DEFAULT_TILE_GRID   = (8, 8)


def enhance_contrast(
    gray: np.ndarray,
    clip_limit: float = DEFAULT_CLIP_LIMIT,
    tile_grid: tuple  = DEFAULT_TILE_GRID,
) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to
    improve local contrast of a grayscale receipt image.

    Thermal receipt paper often produces faded or uneven ink density.
    Global histogram equalization can over-brighten already-bright regions,
    while CLAHE applies equalization locally (tile by tile) and clips the
    amplification to avoid noise explosion — making it ideal for receipts.

    Args:
        gray       (np.ndarray): Grayscale input image.
        clip_limit (float)     : Threshold for contrast limiting.
                                 • 1.0–2.0 → gentle enhancement
                                 • 3.0–4.0 → aggressive (risk of artefacts)
        tile_grid  (tuple)     : Size (rows, cols) of the tile grid.
                                 Smaller tiles → more local adaptation.
                                 (8, 8) works well for most receipt sizes.

    Returns:
        np.ndarray: Contrast-enhanced grayscale image.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    return clahe.apply(gray)