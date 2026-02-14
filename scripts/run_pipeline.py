"""
Script de chay full pipe line (preprocessing + ocr)
"""

import argparse
import os
from src.pipeline import ReceiptOCRPipeline

def main():
    parser = argparse.ArgumentParser(description='Run receipt ocr pipeline')
    parser.add_argument('--input', required=True, help="Input image path")
    parser.add_argument('--output', default='data/results/', help='Output directory')
    parser.add_argument('--save-intermediate', action='store_true', help='save intermediate preprocessing step') # luu anh sau khi preprcessing

    args = parser.parse_args()
    
    # output dir
    os.makedirs(args.output, exist_ok=True)

    # initialize pieline
    pipeline = ReceiptOCRPipeline()

    # process image
    print(f"Processing: {args.input}")
    text = pipeline.process_image(args.input, save_intermediate=args.save_intermediate)

    # save results
    output_file = os.path.join(args.output, 'extracted_text.txt')
    with open(output_file, 'w') as f:
        f.write(text)

    print(f"\nExtracted Text: \n{text}")
    print(f"\nSaved to: {output_file}")

if __name__ == "__main__":
    main()

