import cv2
import numpy as np

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