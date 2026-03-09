import cv2
import numpy as np
from .check_blue_ink import is_blue_ink

def convert_to_grayscale(img: np.ndarray) -> np.ndarray:
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

    if is_blue_ink(img):
        print("    [grayscale] blue-ink receipt detected -- using R+G blend")
        b, g, r = cv2.split(img)
        return cv2.addWeighted(r, 0.5, g, 0.5, 0)

    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
