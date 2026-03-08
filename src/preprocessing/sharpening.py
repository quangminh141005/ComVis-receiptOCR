import cv2
import numy as np

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