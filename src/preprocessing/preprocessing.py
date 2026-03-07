"""
preprocessing.py
----------------
All preprocessing functions combined into a single file.

Functions:
    Grayscale       : to_grayscale()
    Denoising       : denoise_nlm(), denoise_bilateral()
    Contrast        : enhance_contrast()
    Sharpening      : sharpen()
    Binarization    : binarize()
    Morphology      : morphological_cleanup()

Usage:
    from src.preprocessing.preprocessing import (
        to_grayscale,
        denoise_nlm,
        denoise_bilateral,
        enhance_contrast,
        sharpen,
        binarize,
        morphological_cleanup,
    )
"""

import cv2
import numpy as np


# ============================================================================
#  GRAYSCALE CONVERSION
# ============================================================================

BLUE_INK_THRESHOLD = 10   # min blue-red channel diff to detect blue-ink receipt


def _is_blue_ink(img: np.ndarray) -> bool:
    """
    Detect blue-ink receipts by comparing mean Blue vs Red channel.
    Blue-ink receipts (e.g. printed invoices) lose text-background contrast
    when converted with standard BGR->GRAY — this routes them to a better path.
    """
    b, g, r = cv2.split(img)
    return float(np.mean(b)) > float(np.mean(r)) + BLUE_INK_THRESHOLD


def to_grayscale(img: np.ndarray) -> np.ndarray:
    """
    Convert a BGR image to grayscale using the best method for the receipt type.

    Conversion strategy:
      - Already grayscale (2D)  -> return as-is
      - Blue ink detected       -> R*0.5 + G*0.5  (drops blue channel)
                                   maximises text-background contrast
      - Normal black ink        -> standard cv2 BGR->GRAY

    Args:
        img (np.ndarray): BGR or grayscale input image.

    Returns:
        np.ndarray: Grayscale uint8 image, same H x W.
    """
    if len(img.shape) == 2:
        return img                          # already grayscale

    if _is_blue_ink(img):
        print("    [grayscale] blue-ink receipt detected -- using R+G blend")
        b, g, r = cv2.split(img)
        return cv2.addWeighted(r, 0.5, g, 0.5, 0)

    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# ============================================================================
#  DENOISING
# ============================================================================

def denoise_nlm(gray: np.ndarray, h: int = 6) -> np.ndarray:
    """
    Non-Local Means denoising.
    Preserves fine text strokes better than blurs.
    Best for heavy noise or low-resolution scans.

    Args:
        gray (np.ndarray): Grayscale image.
        h    (int)       : Filter strength.
                           5-10  -> light noise / well-exposed images
                           10-20 -> heavier noise / low-res scans
                           >20   -> may blur fine strokes

    Returns:
        np.ndarray: Denoised grayscale image.
    """
    return cv2.fastNlMeansDenoising(gray, h=h)


def denoise_bilateral(
    gray: np.ndarray,
    d: int = 9,
    sigma_color: float = 75,
    sigma_space: float = 75,
) -> np.ndarray:
    """
    Bilateral filter denoising.
    Faster than NLM and excellent at preserving sharp text edges.
    Recommended for most receipt types.

    Args:
        gray        (np.ndarray): Grayscale image.
        d           (int)       : Diameter of pixel neighbourhood.
        sigma_color (float)     : Filter sigma in color space.
                                  Higher -> more colors mixed together.
        sigma_space (float)     : Filter sigma in coordinate space.
                                  Higher -> farther pixels influence each other.

    Returns:
        np.ndarray: Denoised grayscale image.
    """
    return cv2.bilateralFilter(gray, d, sigma_color, sigma_space)


# ============================================================================
#  CONTRAST ENHANCEMENT  (CLAHE)
# ============================================================================

def enhance_contrast(
    gray: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid: tuple = (8, 8),
) -> np.ndarray:
    """
    CLAHE — Contrast Limited Adaptive Histogram Equalization.
    Applies equalization locally (tile by tile) to handle uneven
    lighting and faded thermal-paper ink without amplifying noise.

    Args:
        gray       (np.ndarray): Grayscale image.
        clip_limit (float)     : Contrast amplification limit.
                                 1.0-2.0 -> gentle
                                 3.0-4.0 -> aggressive (risk of artefacts)
        tile_grid  (tuple)     : (rows, cols) tile grid size.
                                 Smaller -> more local adaptation.

    Returns:
        np.ndarray: Contrast-enhanced grayscale image.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    return clahe.apply(gray)


# ============================================================================
#  SHARPENING
# ============================================================================

def sharpen(gray: np.ndarray) -> np.ndarray:
    """
    Laplacian sharpening kernel.
    Enhances text edges — useful for blurry or slightly out-of-focus
    receipt photos taken by phone cameras.

    The 3x3 kernel boosts the centre pixel relative to its neighbours:
        [ 0  -1   0]
        [-1   5  -1]
        [ 0  -1   0]

    Args:
        gray (np.ndarray): Grayscale image.

    Returns:
        np.ndarray: Sharpened grayscale image.
    """
    kernel = np.array(
        [[ 0, -1,  0],
         [-1,  5, -1],
         [ 0, -1,  0]], dtype=np.float32
    )
    return cv2.filter2D(gray, -1, kernel)


# ============================================================================
#  BINARIZATION  (Adaptive Thresholding)
# ============================================================================

def binarize(
    gray: np.ndarray,
    block_size: int = 41,
    c: int = 10,
) -> np.ndarray:
    """
    Adaptive Gaussian Thresholding -> clean binary image.

    Preferred over global Otsu for receipts because thermal-paper images
    have uneven lighting, shadows, or faded regions that a single global
    threshold handles poorly.

    Args:
        gray       (np.ndarray): Grayscale image.
        block_size (int)       : Local neighbourhood window size (must be odd >= 3).
                                 Larger  -> handles uneven lighting / shadows better
                                 Smaller -> more sensitive to local detail
        c          (int)       : Constant subtracted from the local mean.
                                 Higher  -> thinner strokes, less background noise
                                 Lower   -> thicker strokes, more detail preserved

    Returns:
        np.ndarray: Binary (black-and-white) image.
    """
    assert block_size % 2 == 1 and block_size >= 3, \
        "block_size must be an odd integer >= 3"

    return cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        c,
    )


# ============================================================================
#  MORPHOLOGICAL CLEANUP
# ============================================================================

def morphological_cleanup(
    binary: np.ndarray,
    kernel_size: tuple = (1, 1),
) -> np.ndarray:
    """
    Morphological CLOSE then OPEN on a binary receipt image.

    CLOSE : fills tiny gaps / broken strokes inside characters
            (fixes broken letters common in thermal-print receipts)
    OPEN  : removes isolated noise pixels that survived binarization

    Kernel size guide:
        (1, 1) -> minimal effect (safe default)
        (2, 2) -> light gap filling  (good for most receipts)
        (3, 3) -> stronger repair    (for heavily fragmented text)
        (4, 4) -> aggressive         (risk of merging nearby characters)

    Args:
        binary      (np.ndarray): Binary image.
        kernel_size (tuple)     : (h, w) of the structuring element.

    Returns:
        np.ndarray: Morphologically cleaned binary image.
    """
    kernel = np.ones(kernel_size, np.uint8)
    closed  = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN,  kernel)
    return cleaned