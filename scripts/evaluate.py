import os 
import time
import editdistance
from tqdm import tqdm
from src.pipeline import ReceiptOCRPipeline

def load_ground_truth(gt_path: str) -> str:
    """Load ground truth text from CSV file (text is in columns 8 onwards)."""
    lines = []
    try:
        with open(gt_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 8:
                    text = ",".join(parts[8:])
                    lines.append(text.strip())
    except Exception as e:
        print(f"Error reading {gt_path}: {e}")
        return ""
    
    return "\n".join(lines)


def normalize(text: str) -> str:
    """Normalize text for comparison."""
    text = text.lower()
    text = text.replace(" ", "")
    text = text.replace("\n", "")
    return text


def evaluate_folder(folder_path: str):
    """Evaluate OCR pipeline on a folder of images."""
    if not os.path.exists(folder_path):
        raise ValueError(f"Folder not found: {folder_path}")
    
    pipeline = ReceiptOCRPipeline()

    total_distance = 0
    total_chars = 0
    total_samples = 0
    skipped_samples = 0

    start_time = time.time()

    image_files = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    for file in tqdm(image_files, desc="evaluating", unit="images"):
        image_path = os.path.join(folder_path, file)
        
        # Handle all image extensions properly
        base_name = os.path.splitext(file)[0]
        gt_path = os.path.join(folder_path, base_name + ".txt")

        if not os.path.exists(gt_path):
            print(f"Missing GT for {file}")
            skipped_samples += 1
            continue

        try:
            pred = pipeline.process_image(image_path)
        except Exception as e:
            print(f"Error processing {file}: {e}")
            skipped_samples += 1
            continue

        gt = load_ground_truth(gt_path)

        pred = normalize(pred)
        gt = normalize(gt)

        if len(gt) == 0:
            skipped_samples += 1
            continue

        distance = editdistance.eval(pred, gt)

        total_distance += distance
        total_chars += len(gt)
        total_samples += 1

    total_time = time.time() - start_time
    
    # Avoid division by zero
    if total_chars == 0:
        print("\nERROR: No valid ground truth samples found!")
        return
    
    dataset_cer = total_distance / total_chars

    print("\n" + "="*50)
    print("FINAL RESULT")
    print("="*50)
    print(f"Total images found: {len(image_files)}")
    print(f"Valid samples evaluated: {total_samples}")
    print(f"Skipped samples: {skipped_samples}")
    print(f"Dataset CER: {dataset_cer:.4f}")
    print(f"Dataset Accuracy (1-CER): {1 - dataset_cer:.4f}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Avg time per image: {total_time / len(image_files):.4f}s")
    print("="*50)


if __name__ == "__main__":
    evaluate_folder("data/SROIE2019/task1train")