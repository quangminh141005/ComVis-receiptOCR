import cv2
import numpy as np

DEFAULT_KERNEL_SIZE = (2,2)

def morphological_cleanup(binary: np.ndarray, kernel_size=DEFAULT_KERNEL_SIZE):
    kernel = np.ones(kernel_size, np.uint8)
    connected = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(connected, cv2.MORPH_OPEN, kernel)
    return connected

