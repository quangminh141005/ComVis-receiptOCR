import cv2
import numpy as np

DEFAULT_DENOISE_H = 10   # filter strength; higher = more smoothing


def denoise(gray: np.ndarray, h = DEFAULT_DENOISE_H) -> np.ndarray:
    return cv2.fastNlMeansDenoising(gray, h=h)