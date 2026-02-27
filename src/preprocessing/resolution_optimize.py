import cv2
import numpy as np

def convert_to_optimize_resolution(image: np.ndarray, target_width=1000) -> np.ndarray:
    original_height, original_width = image.shape[:2] 

    lower_bound = target_width * 0.9
    upper_bound = target_width * 1.1

    if lower_bound <= original_width <= upper_bound:
        return image
    
    # Create new height corresponding to the aspect ratio
    scaling_factor = target_width / original_width
    new_height = int(original_height * scaling_factor)
    new_dimensions = (target_width, new_height)

    # Choose optimal resizing math
    if original_width < target_width:
        # upscaling if too small
        interpolation_method = cv2.INTER_CUBIC
    else:
        # downscale if image is too massive
        interpolation_method = cv2.INTER_AREA

    optimized_image = cv2.resize(image, new_dimensions, interpolation=interpolation_method)
    
    return optimized_image

