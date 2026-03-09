import os
import time
import csv
import re
import editdistance
from tqdm import tqdm
from src.pipeline import ReceiptOCRPipeline


# ============================================================
# Ground truth loader
# ============================================================

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


# ============================================================
# Text normalization
# ============================================================

def normalize(text: str):

    if text is None:
        return ""

    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()

    return text


# ============================================================
# Word Error Rate
# ============================================================

def compute_wer(pred: str, gt: str):

    pred_words = pred.split()
    gt_words = gt.split()

    dist = editdistance.eval(pred_words, gt_words)

    return dist / max(len(gt_words), 1)


# ============================================================
# Evaluation
# ============================================================

def evaluate_pipelines(folder_path: str):

    if not os.path.exists(folder_path):
        raise ValueError(f"Folder not found: {folder_path}")

    # ============================================================
    # Define pipelines here
    # ============================================================

    pipeline_configs = {

        "no_preprocessing": {
            "preprocessing_steps": [],
            "ocr_config": {"lang": "eng", "psm": 6}
        },

        "bilateral_pipeline": {
            "preprocessing_steps": [
                "grayscale",
                "denoise_bilateral",
                "adaptive_threshold",
                "morphology"
            ],
            "ocr_config": {"lang": "eng", "psm": 6}
        },

        "nlm_pipeline": {
            "preprocessing_steps": [
                "grayscale",
                "denoise_nlm",
                "adaptive_threshold",
                "morphology"
            ],
            "ocr_config": {"lang": "eng", "psm": 6}
        },

        "clahe_pipeline": {
            "preprocessing_steps": [
                "grayscale",
                "denoise_bilateral",
                "contrast",
                "adaptive_threshold",
                "morphology"
            ],
            "ocr_config": {"lang": "eng", "psm": 6}
        }
    }

    # ============================================================
    # Initialize pipelines
    # ============================================================

    pipelines = {
        name: ReceiptOCRPipeline(cfg)
        for name, cfg in pipeline_configs.items()
    }

    # ============================================================
    # Metrics storage
    # ============================================================

    stats = {}

    for name in pipelines:

        stats[name] = {
            "char_distance": 0,
            "char_total": 0,
            "wer_total": 0,
            "exact_match": 0,
            "runtime": 0,
            "images": 0
        }

    improved = 0
    worse = 0
    same = 0

    image_files = sorted([
        f for f in os.listdir(folder_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    start_total = time.time()

    # ============================================================
    # Evaluation loop
    # ============================================================

    for file in tqdm(image_files, desc="Evaluating", unit="img"):

        image_path = os.path.join(folder_path, file)
        base = os.path.splitext(file)[0]
        gt_path = os.path.join(folder_path, base + ".txt")

        if not os.path.exists(gt_path):
            continue

        gt = normalize(load_ground_truth(gt_path))

        if len(gt) == 0:
            continue

        pipeline_cers = {}

        for name, pipeline in pipelines.items():

            start = time.time()

            try:
                pred = pipeline.process_image(image_path)
            except:
                pred = ""

            runtime = time.time() - start

            pred = normalize(pred)

            dist = editdistance.eval(pred, gt)
            cer = dist / len(gt)
            wer = compute_wer(pred, gt)

            stats[name]["char_distance"] += dist
            stats[name]["char_total"] += len(gt)
            stats[name]["wer_total"] += wer
            stats[name]["runtime"] += runtime
            stats[name]["images"] += 1

            if pred == gt:
                stats[name]["exact_match"] += 1

            pipeline_cers[name] = cer

        # ----------------------------------------------------
        # Improvement comparison (vs baseline)
        # ----------------------------------------------------

        baseline = pipeline_cers.get("no_preprocessing")

        for name, cer in pipeline_cers.items():

            if name == "no_preprocessing":
                continue

            if cer < baseline:
                improved += 1
            elif cer > baseline:
                worse += 1
            else:
                same += 1

    total_time = time.time() - start_total

    # ============================================================
    # Compute final metrics
    # ============================================================

    results = []

    for name in pipelines:

        s = stats[name]

        cer = s["char_distance"] / max(s["char_total"], 1)
        wer = s["wer_total"] / max(s["images"], 1)
        exact = s["exact_match"] / max(s["images"], 1)
        runtime = s["runtime"] / max(s["images"], 1)

        results.append({
            "pipeline": name,
            "CER": cer,
            "WER": wer,
            "ExactMatch": exact,
            "AvgRuntime": runtime
        })

    results = sorted(results, key=lambda x: x["CER"])

    # ============================================================
    # Print results
    # ============================================================

    print("\n" + "=" * 70)
    print("OCR PIPELINE EVALUATION")
    print("=" * 70)

    print(f"Total images: {len(image_files)}")

    print("\nPipeline Results")

    for r in results:

        print(f"\nPipeline: {r['pipeline']}")
        print(f"  CER          : {r['CER']:.4f}")
        print(f"  WER          : {r['WER']:.4f}")
        print(f"  Exact Match  : {r['ExactMatch']:.4f}")
        print(f"  Avg Runtime  : {r['AvgRuntime']:.4f}s")

    print("\nImage level comparison (vs baseline)")
    print(f"Improved images : {improved}")
    print(f"Worse images    : {worse}")
    print(f"Same            : {same}")

    print("\nTotal evaluation time:", f"{total_time:.2f}s")

    print("=" * 70)

    # ============================================================
    # Save results to CSV
    # ============================================================

    output_file = "pipeline_results.csv"

    with open(output_file, "w", newline="") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=["pipeline", "CER", "WER", "ExactMatch", "AvgRuntime"]
        )

        writer.writeheader()

        for r in results:
            writer.writerow(r)

    print(f"\nResults saved to: {output_file}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    evaluate_pipelines("data/SROIE2019/Stage1train")