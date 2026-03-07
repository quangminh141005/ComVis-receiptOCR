import os
import time
import csv
import re
import editdistance
from tqdm import tqdm
from src.pipeline import ReceiptOCRPipeline


def load_ground_truth(gt_path: str):
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


def normalize(text: str):
    if text is None:
        return ""

    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()

    return text


def evaluate_compare(folder_path: str):

    if not os.path.exists(folder_path):
        raise ValueError(f"Folder not found: {folder_path}")

    # -------------------------
    # Pipeline configurations
    # -------------------------

    no_pre_config = {
        "preprocessing_steps": [],
        "ocr_config": {"lang": "eng", "psm": 6}
    }

    pre_config = {
        "preprocessing_steps": [
            "resolution",
            "grayscale",
            "denoise",
            "contrast",
            "adaptive_threshold",
            "flip",
            "morphology",
            "flip"
        ],
        "ocr_config": {"lang": "eng", "psm": 6}
    }

    pipeline_no_pre = ReceiptOCRPipeline(no_pre_config)
    pipeline_pre = ReceiptOCRPipeline(pre_config)

    image_files = sorted([
        f for f in os.listdir(folder_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    # -------------------------
    # Metrics storage
    # -------------------------

    total_distance_no = 0
    total_chars_no = 0

    total_distance_pre = 0
    total_chars_pre = 0

    improved = 0
    worse = 0
    same = 0

    start_time_no = time.time()

    # -------------------------
    # Evaluation loop
    # -------------------------

    for file in tqdm(image_files, desc="Evaluating", unit="img"):

        image_path = os.path.join(folder_path, file)
        base = os.path.splitext(file)[0]
        gt_path = os.path.join(folder_path, base + ".txt")

        if not os.path.exists(gt_path):
            continue

        gt = load_ground_truth(gt_path)
        gt = normalize(gt)

        if len(gt) == 0:
            continue

        # -------- NO PREPROCESS --------

        try:
            pred_no = pipeline_no_pre.process_image(image_path)
        except:
            pred_no = ""

        pred_no = normalize(pred_no)

        dist_no = editdistance.eval(pred_no, gt)
        cer_no = dist_no / len(gt)

        total_distance_no += dist_no
        total_chars_no += len(gt)

        # -------- WITH PREPROCESS --------

        try:
            pred_pre = pipeline_pre.process_image(image_path)
        except:
            pred_pre = ""

        pred_pre = normalize(pred_pre)

        dist_pre = editdistance.eval(pred_pre, gt)
        cer_pre = dist_pre / len(gt)

        total_distance_pre += dist_pre
        total_chars_pre += len(gt)

        # -------- compare improvement --------

        if cer_pre < cer_no:
            improved += 1
        elif cer_pre > cer_no:
            worse += 1
        else:
            same += 1

    total_time = time.time() - start_time_no

    cer_no = total_distance_no / total_chars_no
    cer_pre = total_distance_pre / total_chars_pre

    improvement = cer_no - cer_pre

    # -------------------------
    # Results
    # -------------------------

    print("\n" + "=" * 60)
    print("OCR PREPROCESSING COMPARISON")
    print("=" * 60)

    print(f"Total images: {len(image_files)}")

    print("\nCER comparison")
    print(f"No preprocessing CER : {cer_no:.4f}")
    print(f"With preprocessing CER: {cer_pre:.4f}")

    print(f"\nCER improvement: {improvement:.4f}")

    print("\nImage level comparison")
    print(f"Improved images : {improved}")
    print(f"Worse images    : {worse}")
    print(f"Same            : {same}")

    print("\nSpeed")
    print(f"Total time: {total_time:.2f}s")
    print(f"Avg time per image (2 pipelines): {total_time / len(image_files):.4f}s")

    print("=" * 60)


if __name__ == "__main__":
    evaluate_compare("data/SROIE2019/task1train")