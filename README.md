# ComVis-receiptOCR

computer vision project for extracting text from receipt image

## setup

### prequisites
- python 3.10
- Tesseract OCR install in your OS (`sudo apt install tesseract-ocr` with ubuntu/debian - window thi khong biet dau nhe:D)

### installation
```bash

# clone
git clone git@github.com:quangminh141005/ComVis-receiptOCR.git
cd Comvis-receiptOCR.git

# install dependencies
pip install -r requirements.txt

```

## usage

### run from cli
```bash
python scripts/run_pipeline.py --input /ComVis-receiptOCR/data/SROIE2019/0325updated.task1train(626p)/pick-your-self-a-receipt-image-ok.jpg" --output data/results/
```