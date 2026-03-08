import cv2
import numpy as np

def check_contrast(image_path):

    image = cv2.imread(image_path)

    # convert to grayscale if needed
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # ensure uint8
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    image = image.astype(np.float32)

    # Basic statistics
    mean_intensity = np.mean(image)
    std_dev = np.std(image)  # global contrast (most common metric)

    # RMS contrast (same as std for grayscale)
    rms_contrast = np.sqrt(np.mean((image - mean_intensity) ** 2))

    # Michelson contrast
    min_val = np.min(image)
    max_val = np.max(image)
    if (max_val + min_val) != 0:
        michelson = (max_val - min_val) / (max_val + min_val)
    else:
        michelson = 0

    print("Mean intensity:", mean_intensity)
    print("Standard deviation (global contrast):", std_dev)
    print("RMS contrast:", rms_contrast)
    print("Michelson contrast:", michelson)
    print("Min intensity:", min_val)
    print("Max intensity:", max_val)

if __name__ == "__main__":
    check_contrast("data/SROIE2019/task1train/X51005301661.jpg")