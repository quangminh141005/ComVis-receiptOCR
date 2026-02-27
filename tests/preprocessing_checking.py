import os
import cv2
import numpy as np
from typing import Dict, Optional

# Assuming your imports work perfectly:
from src.preprocessing.grayscale import convert_to_grayscale
from src.preprocessing.thresholding import apply_otsu_threshold
from src.preprocessing.resolution_optimize import convert_to_optimize_resolution

class TestingPipeline:
    def __init__(self, output_dir: str = "pipeline_tests", config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.output_dir = output_dir
    
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"Created output directory: {self.output_dir}")
    
    def _default_config(self) -> Dict:
        return {
            'preprocessing_steps': ['resolution', 'grayscale', 'otsu'],
        }

    def process_image(self, image_path: str) -> np.ndarray:
        # 1. Load the image
        image = cv2.imread(image_path)
        if image is None: 
            raise ValueError(f"Cannot load image at {image_path}")
        
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        print(f"\n--- Processing: {base_name} ---")

        # Save the original image for comparison
        orig_path = os.path.join(self.output_dir, f"{base_name}_0_original.jpg")
        cv2.imwrite(orig_path, image)

        processed = image.copy()

        for i, step in enumerate(self.config['preprocessing_steps'], start=1):
            if step == 'grayscale':
                processed = convert_to_grayscale(processed)
            elif step == 'otsu':
                processed = apply_otsu_threshold(processed)
            elif step == 'resolution':
                processed = convert_to_optimize_resolution(processed)
            else:
                print(f"Warning: Unknown step '{step}' skipped.")
                continue
            
            step_filename = f"{base_name}_{i}_{step}.jpg"
            save_path = os.path.join(self.output_dir, step_filename)
            cv2.imwrite(save_path, processed)
            print(f"Saved step {i}: {step_filename}")
        
        return processed


if __name__ == "__main__":
    # You can customize the sequence by passing a config dictionary
    custom_config = {
        'preprocessing_steps': ['resolution'] 
    }
    
    # Initialize the pipeline
    pipeline = TestingPipeline(output_dir="data/processed", config=custom_config)
    
    test_image_path = "data/SROIE2019/task1train/X51005301661.jpg" 
    
    try:
        final_image = pipeline.process_image(test_image_path)
        print("Pipeline finished successfully in data/processed")
    except Exception as e:
        print(f"An error occurred: {e}")