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
    except Exception as e:
        print(f"Error loading ground truth {gt_path}: {e}")
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
# Intelligent pipeline validation
# ============================================================

def is_valid_pipeline(steps):
    """
    Filter out invalid/redundant pipeline combinations based on
    domain knowledge about OCR preprocessing.
    
    Returns True if pipeline is valid, False if it should be filtered out.
    """
    
    # Rule 1: Don't use both denoising methods together (redundant)
    if "denoise_bilateral" in steps and "denoise_nlm" in steps:
        return False
    
    # Rule 2: Don't use both thresholding methods together (conflicting)
    if "adaptive_threshold" in steps and "otsu_threshold" in steps:
        return False
    
    # Rule 3: Threshold should come BEFORE morphology (logical order)
    # Morphology is meant to clean up after thresholding
    threshold_methods = ["adaptive_threshold", "otsu_threshold"]
    has_threshold = any(t in steps for t in threshold_methods)
    has_morphology = "morphology" in steps
    
    if has_threshold and has_morphology:
        threshold_idx = next(i for i, s in enumerate(steps) if s in threshold_methods)
        morphology_idx = steps.index("morphology")
        if threshold_idx > morphology_idx:
            return False
    
    # Rule 4: Resolution enhancement should be early in pipeline
    # Upscaling should happen before major transformations
    if "resolution" in steps:
        res_idx = steps.index("resolution")
        
        # Resolution should come before threshold
        if has_threshold:
            threshold_idx = next(i for i, s in enumerate(steps) if s in threshold_methods)
            if res_idx > threshold_idx:
                return False
        
        # Resolution should come before morphology
        if has_morphology:
            morphology_idx = steps.index("morphology")
            if res_idx > morphology_idx:
                return False
    
    # Rule 5: Sharpening should be later in pipeline
    # Sharpening on raw/noisy images is counterproductive
    if "sharpening" in steps:
        sharp_idx = steps.index("sharpening")
        
        # Sharpening shouldn't be in the first 2 positions
        # (grayscale is position 0, so sharp_idx < 2 means position 1 or 2)
        if sharp_idx < 2 and len(steps) > 3:
            return False
        
        # If denoising exists, sharpening should come after it
        denoise_methods = ["denoise_bilateral", "denoise_nlm"]
        has_denoise = any(d in steps for d in denoise_methods)
        if has_denoise:
            denoise_idx = next(i for i, s in enumerate(steps) if s in denoise_methods)
            if sharp_idx < denoise_idx:
                return False
    
    # Rule 6: Denoising should come early (before threshold)
    denoise_methods = ["denoise_bilateral", "denoise_nlm"]
    has_denoise = any(d in steps for d in denoise_methods)
    
    if has_denoise and has_threshold:
        denoise_idx = next(i for i, s in enumerate(steps) if s in denoise_methods)
        threshold_idx = next(i for i, s in enumerate(steps) if s in threshold_methods)
        
        # Denoise should come before threshold
        if denoise_idx > threshold_idx:
            return False
    
    return True


# ============================================================
# Generate pipeline combinations with INTELLIGENT FILTERING
# ============================================================

def generate_pipelines(min_steps=1, max_steps=5, max_pipelines=None, use_order=True):
    """
    Generate pipeline configurations with intelligent filtering.
    
    Args:
        min_steps: Minimum number of optional steps (default: 1)
        max_steps: Maximum number of optional steps (default: 5)
        max_pipelines: Maximum total pipelines to generate (default: None = unlimited)
        use_order: If True, use permutations (order matters), else combinations
    
    Returns:
        Dictionary of pipeline configurations
    """
    
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
    filtered_count = 0
    total_generated = 0

    # Clamp max_steps to available steps
    max_steps = min(max_steps, len(optional_steps))
    
    print(f"\n{'='*60}")
    print(f"Pipeline Generation Configuration:")
    print(f"{'='*60}")
    print(f"  Base steps: {base_steps}")
    print(f"  Optional steps: {len(optional_steps)}")
    print(f"  Min steps: {min_steps}")
    print(f"  Max steps: {max_steps}")
    print(f"  Use order (permutations): {use_order}")
    print(f"  Intelligent filtering: ENABLED")
    print(f"  Max pipelines limit: {max_pipelines if max_pipelines else 'Unlimited'}")
    print(f"{'='*60}\n")

    # Use permutations if order matters, combinations otherwise
    iterator_func = itertools.permutations if use_order else itertools.combinations

    print("Generating and filtering pipelines...")
    
    for r in range(min_steps, max_steps + 1):
        
        for combo in iterator_func(optional_steps, r):
            
            total_generated += 1
            steps = base_steps + list(combo)
            
            # Apply intelligent filtering
            if not is_valid_pipeline(steps):
                filtered_count += 1
                continue
            
            # Check if we've hit the pipeline limit
            if max_pipelines and pipeline_id >= max_pipelines:
                print(f"\n✓ Reached maximum pipeline limit: {max_pipelines}")
                print(f"✓ Generated {total_generated} total combinations")
                print(f"✓ Filtered out {filtered_count} invalid pipelines ({filtered_count/total_generated*100:.1f}%)")
                print(f"✓ Kept {len(pipelines)} valid pipelines ({len(pipelines)/total_generated*100:.1f}%)")
                return pipelines

            name = f"pipe_{pipeline_id}"

            pipelines[name] = {
                "preprocessing_steps": steps,
                "ocr_config": {"lang": "eng", "psm": 6}
            }

            pipeline_id += 1

    print(f"\Generated {total_generated} total combinations")
    print(f"✓ Filtered out {filtered_count} invalid pipelines ({filtered_count/total_generated*100:.1f}%)")
    print(f"✓ Kept {len(pipelines)} valid pipelines ({len(pipelines)/total_generated*100:.1f}%)\n")
    
    return pipelines


# ============================================================
# Worker task
# ============================================================

def process_image(args):
    image_path, gt = args

    start = time.time()

    try:
        pred = pipeline.process_image(image_path)
    except Exception as e:
        print(f"Error processing {os.path.basename(image_path)}: {e}")
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

def evaluate(folder_path, min_steps=1, max_steps=5, max_pipelines=None, 
             use_order=True, num_workers=6):
    """
    Evaluate multiple pipeline configurations
    
    Args:
        folder_path: Path to dataset folder
        min_steps: Minimum number of preprocessing steps (default: 1)
        max_steps: Maximum number of preprocessing steps (default: 5)
        max_pipelines: Maximum number of pipelines to test (default: None)
        use_order: Whether step order matters - uses permutations if True (default: True)
        num_workers: Number of parallel workers (default: 6)
    """

    pipeline_configs = generate_pipelines(
        min_steps=min_steps,
        max_steps=max_steps,
        max_pipelines=max_pipelines,
        use_order=use_order
    )

    print(f"Total pipelines to evaluate: {len(pipeline_configs)}")

    # Load dataset
    image_files = sorted([
        f for f in os.listdir(folder_path)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ])

    dataset = []

    print("Loading dataset...")
    for file in tqdm(image_files, desc="Loading images"):

        image_path = os.path.join(folder_path, file)
        base = os.path.splitext(file)[0]
        gt_path = os.path.join(folder_path, base + ".txt")

        if not os.path.exists(gt_path):
            continue

        gt = normalize(load_ground_truth(gt_path))

        if len(gt) == 0:
            continue

        dataset.append((image_path, gt))

    print(f"\nValid images: {len(dataset)}")
    
    # Estimate time
    estimated_time_minutes = len(pipeline_configs) * len(dataset) * 0.5 / num_workers / 60
    if estimated_time_minutes < 60:
        print(f"Estimated total time: ~{estimated_time_minutes:.1f} minutes")
    else:
        print(f"Estimated total time: ~{estimated_time_minutes/60:.1f} hours")
    
    print(f"\n{'='*60}\n")

    results_table = []

    total_start = time.time()

    # ============================================================
    # Evaluate pipeline by pipeline with PROGRESS INDICATOR
    # ============================================================

    pipeline_items = list(pipeline_configs.items())
    
    # Create pipeline-level progress bar
    pipeline_progress = tqdm(
        pipeline_items,
        desc="Overall Progress",
        unit="pipeline",
        position=0,
        colour="green",
        ncols=100
    )

    for idx, (name, config) in enumerate(pipeline_progress, 1):

        # Update the progress bar description with current pipeline
        pipeline_progress.set_description(
            f"Overall Progress [{idx}/{len(pipeline_items)}]"
        )
        
        print(f"\n{'─'*60}")
        print(f"Pipeline: {name} ({idx}/{len(pipeline_items)})")
        print(f"Steps: {' → '.join(config['preprocessing_steps'])}")
        print(f"{'─'*60}")

        char_dist = 0
        char_total = 0
        wer_total = 0
        runtime_total = 0

        start = time.time()

        with Pool(
            num_workers,
            initializer=init_worker,
            initargs=(config,)
        ) as pool:

            results = list(
                tqdm(
                    pool.imap(process_image, dataset),
                    total=len(dataset),
                    desc=f"  Processing images",
                    unit="img",
                    position=1,
                    leave=False,
                    colour="blue",
                    ncols=100
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
            "steps": " → ".join(config["preprocessing_steps"]),
            "CER": cer,
            "WER": wer,
            "Runtime": runtime
        })

        # Show current results
        print(f"  CER: {cer:.4f} | WER: {wer:.4f} | Runtime: {runtime:.3f}s")
        
        # Show current best
        current_best = min(results_table, key=lambda x: x["CER"])
        if current_best["pipeline"] == name:
            print(f"NEW BEST! (CER: {cer:.4f})")

    total_time = time.time() - total_start

    # Sort by CER (best first)
    results_table = sorted(results_table, key=lambda x: x["CER"])

    print(f"\n{'='*60}")
    print(f"EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"\nTop 10 Pipelines by CER:\n")

    for i, r in enumerate(results_table[:10], 1):
        print(f"{i}. {r['pipeline']}")
        print(f"   Steps: {r['steps']}")
        print(f"   CER: {r['CER']:.4f} | WER: {r['WER']:.4f} | Runtime: {r['Runtime']:.3f}s")
        print()

    # Save detailed results
    output_file = "pipeline_parallel_results.csv"
    with open(output_file, "w", newline="", encoding="utf-8") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=["pipeline", "steps", "CER", "WER", "Runtime"]
        )

        writer.writeheader()

        for r in results_table:
            writer.writerow(r)

    # Save summary
    summary_file = "pipeline_summary.txt"
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("="*60 + "\n")
        f.write("PIPELINE EVALUATION SUMMARY\n")
        f.write("="*60 + "\n\n")
        f.write(f"Dataset: {folder_path}\n")
        f.write(f"Images evaluated: {len(dataset)}\n")
        f.write(f"Pipelines tested: {len(pipeline_configs)}\n")
        f.write(f"Total time: {total_time / 60:.2f} minutes\n\n")
        f.write("="*60 + "\n")
        f.write("TOP 10 PIPELINES\n")
        f.write("="*60 + "\n\n")
        
        for i, r in enumerate(results_table[:10], 1):
            f.write(f"{i}. {r['pipeline']}\n")
            f.write(f"   Steps: {r['steps']}\n")
            f.write(f"   CER: {r['CER']:.4f} | WER: {r['WER']:.4f} | Runtime: {r['Runtime']:.3f}s\n\n")

    print(f"✓ Results saved to {output_file}")
    print(f"✓ Summary saved to {summary_file}")
    print(f"✓ Total evaluation time: {total_time / 60:.2f} minutes ({total_time:.1f} seconds)")
    print(f"✓ Evaluated {len(pipeline_configs)} pipelines on {len(dataset)} images")
    print(f"\n{'='*60}\n")


# ============================================================
# Entry
# ============================================================

if __name__ == "__main__":

    # ============================================================
    # CONFIGURATION - Adjust these parameters
    # ============================================================
    
    DATA_PATH = "data/SROIE2019/Stage1train"
    
    # Pipeline generation settings
    MIN_STEPS = 2        # Minimum preprocessing steps (excluding base grayscale)
    MAX_STEPS = 5        # Maximum preprocessing steps (recommended: 4-5)
    MAX_PIPELINES = 500  # Maximum total pipelines (safety cap, use None for unlimited)
    USE_ORDER = True     # True = order matters (permutations), False = combinations only
    
    # Execution settings
    NUM_WORKERS = 6      # Parallel workers (adjust based on CPU cores)
    
    # ============================================================
    # Expected results with these settings:
    # - max_steps=5 with filtering: ~400-600 valid pipelines
    # - Evaluation time: 2-3 hours (depending on hardware)
    # - Should find optimal 4-5 step pipelines
    # ============================================================
    
    evaluate(
        folder_path=DATA_PATH,
        min_steps=MIN_STEPS,
        max_steps=MAX_STEPS,
        max_pipelines=MAX_PIPELINES,
        use_order=USE_ORDER,
        num_workers=NUM_WORKERS
    )