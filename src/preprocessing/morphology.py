import cv2
import numpy as np

def morphological_cleanup(binary, kernel_size=(2, 2)):
    kernel = np.ones(kernel_size, np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    return cleaned
