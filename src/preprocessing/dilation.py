import cv2
import numpy as np

DEFAULT_DILATION_KERNEL = (3,3)

def dilate(binary: np.ndarray, kernel_size=DEFAULT_DILATION_KERNEL, iterations=1) -> np.ndarray:
    kernel = np.ones(kernel_size, np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=iterations)
    return dilated