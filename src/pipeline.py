"""
main pipeline
"""

import cv2
import numpy as np
from typing import Dict, Optional
from src.preprocessing.grayscale import convert_to_grayscale
from src.preprocessing.thresholding import apply_otsu_threshold
from src.ocr.tesseract_ocr import TesseractOCR

class ReceiptOCRPipeline:
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize pipeline with configuration
        
        Args:
            config: Dictionary with pipeline configuration
                   - preprocessing_steps: list of preprocessing methods
                   - ocr_config: OCR engine configuration
        """
        self.config = config or self._default_config()
        self.ocr_engine = TesseractOCR(self.config.get('ocr_config', {}))

    def _default_config(self) -> Dict:
        return {
            'preprocessing_steps': ['grayscale', 'otsu'],
            'ocr_config': {
                'lang': 'eng',
                'psm': 6 # page segmentation mode (6 la segment thanh tung block)
            }
        }
    
    def process_image(self, image_path: str, save_intermediate:bool = False) -> str:
        """
        process a single receipt

        args: 
            image_path
            save_intermediate

        return:
            extracted text from receipt
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Khong load duoc anh roi, path: {image_path}")
        
        # apply preprocessing
        processed_image = self._preprocess(image)

        if save_intermediate:
            self._save_intermediate(image_path, processed_image)

        # apply ocr
        text = self.ocr_engine.extract_text(processed_image)

        return text
    
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        apply preprocessing step
        """
        processed = image.copy()

        for step in self.config['preprocessing_steps']:
            if step == 'grayscale':
                processed = convert_to_grayscale(processed)
            elif step == 'otsu':
                processed = apply_otsu_threshold(processed)
            # Them method preprocessing o day nha
        
        return processed
    
    def _save_intermediate(self, original_path: str, processed_image: np.ndarray):
        """Luu anh sau khi preprocessing"""
        import os
        filename = os.path.basename(original_path)

        output_dir = "data/processed"

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"create dir {output_dir}")
        
        
        output_path = os.path.join(output_dir, filename)

        cv2.imwrite(output_path, processed_image)
        print(f"save image in {output_path}")
