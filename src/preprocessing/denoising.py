import cv2
import numpy as np

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