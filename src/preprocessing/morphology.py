import cv2
import numpy as np

DEFAULT_KERNEL_SIZE = (3, 3)

def morphological_cleanup(binary: np.ndarray, kernel_size=(2,2)):
    kernel = np.ones(kernel_size, np.uint8)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    return cleaned

