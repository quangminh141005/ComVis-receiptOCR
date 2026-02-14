import pytesseract
import numpy as np
from typing import Dict, Optional

class TesseractOCR:
    """
    Warpper cho engine cua tesseract
    """
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Tesseract OCR
        
        Args:
            config: OCR configuration
                   - lang: Language code (default: 'eng')
                   - psm: Page segmentation mode (default: 6)
                   - oem: OCR Engine mode (default: 3)
        """
        self.config = config or {}
        self.lang = self.config.get('lang', 'eng')
        self.psm = self.config.get('psm', 6)
        self.oem = self.config.get('oem', 3)

    def extract_text(self, image: np.ndarray) -> str:
        custom_config = f'--oem {self.oem} --psm {self.psm}'
        text = pytesseract.image_to_string(
            image,
            lang=self.lang,
            config=custom_config
        )
        return text.strip()
    
    def extract_data(self, image:np.ndarray) -> Dict:
        """
        trả về bounding boxes, confidence score, string
        """
        data = pytesseract.image_to_data(
            image,
            lang=self.lang,
            output_type=pytesseract.Output.DICT
        )
        return data