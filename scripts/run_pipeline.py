"""
Script de chay full pipe line (preprocessing + ocr)
"""

import argparse
import os
from src.pipeline import ReceiptOCRPipeline

def main():
    parser = argparse.ArgumentParser(description='Run receipt ocr pipeline')
    parser.add_argument('--input', require=True, help="Input image path")
    parser.add_argument('--output', default='data/results/', help='Output directory')
    parser.add_argument('--same-intermediate', action='store_true', help='save intermediate preprocessing step') # luu anh sau khi preprcessing

    args = parser_args()
    
    # output dir
    os.makedirs(args.output, exist_ok=True)
    
     