import os
import time
import csv
import re
import itertools
import editdistance
from multiprocessing import Pool
from tqdm import tqdm
from src.pipeline import ReceiptOCRPipeline


# ============================================================
# Global pipeline instance (per worker)
# ============================================================

pipeline = None


# ============================================================
# Worker initializer (runs once per worker)
# ============================================================

def init_worker(config):

    global pipeline
    pipeline = ReceiptOCRPipeline(config)


# ============================================================
# Load ground truth
# ============================================================

def load_ground_truth(gt_path):

    lines = []

    try:
        with open(gt_path, encoding="utf-8") as f:
            reader = csv.reader(f)

            for row in reader:
                if len(row) >= 9:
                    text = ",".join(row[8:])
                    lines.append(text.strip())
    except:
        return ""

    return "\n".join(lines)


# ============================================================
# Normalize text
# ============================================================

def normalize(text):

    if text is None:
        return ""

    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    return text


# ============================================================
# Word Error Rate
# ============================================================

def compute_wer(pred, gt):

    pred_words = pred.split()
    gt_words = gt.split()

    dist = editdistance.eval(pred_words, gt_words)

    return dist / max(len(gt_words), 1)


# ============================================================
# Generate pipeline combinations
# ============================================================

def generate_pipelines():

    base_steps = ["grayscale"]

    optional_steps = [
        "denoise_bilateral",
        "denoise_nlm",
        "resolution",
        "adaptive_threshold",
        "otsu_threshold",
        "morphology",
        "sharpening"
    ]

    pipelines = {}

    pipeline_id = 0

    for r in range(1, len(optional_steps) + 1):

        for combo in itertools.combinations(optional_steps, r):

            steps = base_steps + list(combo)

            name = f"pipe_{pipeline_id}"

            pipelines[name] = {
                "preprocessing_steps": steps,
                "ocr_config": {"lang": "eng", "psm": 6}
            }

            pipeline_id += 1

    return pipelines


# ============================================================
# Worker task
# ============================================================

def process_image(args):

    image_path, gt = args

    start = time.time()

    try:
        pred = pipeline.process_image(image_path)
    except:
        pred = ""

    runtime = time.time() - start

    pred = normalize(pred)

    dist = editdistance.eval(pred, gt)
    cer = dist / max(len(gt), 1)
    wer = compute_wer(pred, gt)

    return dist, len(gt), wer, runtime


# ============================================================
# Main evaluation
# ============================================================

def evaluate(folder_path):

    pipeline_configs = generate_pipelines()

    print("Total pipelines:", len(pipeline_configs))

    image_files = sorted([
        f for f in os.listdir(folder_path)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ])

    dataset = []

    for file in image_files:

        image_path = os.path.join(folder_path, file)
        base = os.path.splitext(file)[0]
        gt_path = os.path.join(folder_path, base + ".txt")

        if not os.path.exists(gt_path):
            continue

        gt = normalize(load_ground_truth(gt_path))

        if len(gt) == 0:
            continue

        dataset.append((image_path, gt))

    print("Valid images:", len(dataset))

    results_table = []

    total_start = time.time()

    # ============================================================
    # Evaluate pipeline by pipeline
    # ============================================================

    for name, config in pipeline_configs.items():

        print("\nEvaluating:", name)
        print("Steps:", config["preprocessing_steps"])

        char_dist = 0
        char_total = 0
        wer_total = 0
        runtime_total = 0

        start = time.time()

        with Pool(
            6,
            initializer=init_worker,
            initargs=(config,)
        ) as pool:

            results = list(
                tqdm(
                    pool.imap(process_image, dataset),
                    total=len(dataset)
                )
            )

        for dist, char_len, wer, runtime in results:

            char_dist += dist
            char_total += char_len
            wer_total += wer
            runtime_total += runtime

        cer = char_dist / max(char_total, 1)
        wer = wer_total / max(len(dataset), 1)
        runtime = runtime_total / max(len(dataset), 1)

        results_table.append({
            "pipeline": name,
            "steps": config["preprocessing_steps"],
            "CER": cer,
            "WER": wer,
            "Runtime": runtime
        })

        print("CER:", round(cer, 4))
        print("WER:", round(wer, 4))
        print("Runtime:", round(runtime, 3))

    total_time = time.time() - total_start

    results_table = sorted(results_table, key=lambda x: x["CER"])

    print("\nTop pipelines:\n")

    for r in results_table[:10]:

        print(r["pipeline"])
        print("steps:", r["steps"])
        print("CER:", round(r["CER"], 4))
        print("WER:", round(r["WER"], 4))
        print("Runtime:", round(r["Runtime"], 3))
        print()

    with open("pipeline_parallel_results.csv", "w", newline="") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=["pipeline", "steps", "CER", "WER", "Runtime"]
        )

        writer.writeheader()

        for r in results_table:
            writer.writerow(r)

    print("\nResults saved to pipeline_parallel_results.csv")
    print("Total evaluation time:", round(total_time, 2), "seconds")


# ============================================================
# Entry
# ============================================================

if __name__ == "__main__":

    evaluate("data/SROIE2019/Stage1train")