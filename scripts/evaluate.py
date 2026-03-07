import os
import time
import csv
import re
import editdistance
from tqdm import tqdm
from src.pipeline import ReceiptOCRPipeline


def load_ground_truth(gt_path: str) -> str:
    """
    Load ground truth text from SROIE-style annotation files.
    Text content starts from column 8 onward.
    Uses CSV parser to correctly handle commas in text.
    """
    lines = []

    try:
        with open(gt_path, encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 9:
                    text = ",".join(row[8:])
                    lines.append(text.strip())
    except Exception as e:
        print(f"Error reading {gt_path}: {e}")
        return ""

    return "\n".join(lines)


def normalize(text: str) -> str:
    """
    Normalize text for fair OCR comparison.
    """
    if text is None:
        return ""

    text = text.lower()
    text = re.sub(r'\s+', ' ', text)  # collapse whitespace
    text = text.strip()

    return text


def compute_wer(pred: str, gt: str) -> float:
    """
    Compute Word Error Rate.
    """
    pred_words = pred.split()
    gt_words = gt.split()

    if len(gt_words) == 0:
        return 0

    distance = editdistance.eval(pred_words, gt_words)
    return distance / len(gt_words)


def evaluate_folder(folder_path: str):

    if not os.path.exists(folder_path):
        raise ValueError(f"Folder not found: {folder_path}")

    pipeline = ReceiptOCRPipeline()

    total_distance = 0
    total_chars = 0
    total_words = 0
    total_word_distance = 0

    total_samples = 0
    skipped_samples = 0
    exact_matches = 0

    start_time = time.time()

    image_files = sorted([
        f for f in os.listdir(folder_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    worst_samples = []

    for file in tqdm(image_files, desc="Evaluating", unit="img"):

        image_path = os.path.join(folder_path, file)
        base_name = os.path.splitext(file)[0]
        gt_path = os.path.join(folder_path, base_name + ".txt")

        if not os.path.exists(gt_path):
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

        char_distance = editdistance.eval(pred, gt)
        cer = char_distance / len(gt)

        word_distance = editdistance.eval(pred.split(), gt.split())
        gt_words = len(gt.split())
        wer = word_distance / gt_words if gt_words > 0 else 0

        total_distance += char_distance
        total_chars += len(gt)

        total_word_distance += word_distance
        total_words += gt_words

        if pred == gt:
            exact_matches += 1

        worst_samples.append((file, cer))

        total_samples += 1

    total_time = time.time() - start_time

    if total_chars == 0:
        print("No valid ground truth samples.")
        return

    dataset_cer = total_distance / total_chars
    dataset_wer = total_word_distance / total_words if total_words > 0 else 0
    exact_match_rate = exact_matches / total_samples if total_samples > 0 else 0

    worst_samples.sort(key=lambda x: x[1], reverse=True)
    worst_samples = worst_samples[:10]

    print("\n" + "="*60)
    print("OCR EVALUATION RESULT")
    print("="*60)

    print(f"Images found: {len(image_files)}")
    print(f"Valid samples: {total_samples}")
    print(f"Skipped samples: {skipped_samples}")

    print("\nAccuracy metrics")
    print(f"CER (Character Error Rate): {dataset_cer:.4f}")
    print(f"WER (Word Error Rate):      {dataset_wer:.4f}")
    print(f"Exact Match Rate:           {exact_match_rate:.4f}")

    print("\nSpeed metrics")
    print(f"Total time: {total_time:.2f}s")
    print(f"Avg time per image: {total_time / total_samples:.4f}s")

    print("\nWorst samples")
    for name, cer in worst_samples:
        print(f"{name:30} CER={cer:.3f}")

    print("="*60)


if __name__ == "__main__":
    evaluate_folder("data/SROIE2019/task1train")