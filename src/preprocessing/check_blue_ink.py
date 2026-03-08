import cv2
import numpy as np

BLUE_INK_THRESHOLD = 10 

def _is_blue_ink(img: np.ndarray) -> bool:
    """
    Detect blue-ink receipts by comparing mean Blue vs Red channel.
    Blue-ink receipts (e.g. printed invoices) lose text-background contrast
    when converted with standard BGR->GRAY — this routes them to a better path.
    """
    b, g, r = cv2.split(img)
    return float(np.mean(b)) > float(np.mean(r)) + BLUE_INK_THRESHOLD