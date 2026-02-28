import cv2
import numpy as np

def contrast_clahe(image: np.ndarray):
    # create CLAHE object
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    enhanced = clahe.apply(image)

    return enhanced