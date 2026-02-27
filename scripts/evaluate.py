import os 
import time
import editdistance
from tqdm import tqdm
from src.pipeline import ReceiptOCRPipeline

# select only text, ignore coords
def load_ground_truth(gt_path: str) -> str:
    lines = []
    with open(gt_path, 'r', encoding='utf-8') as f:
        for line in f: 
            parts = line.strip().split(',')
            text = ",".join(parts[8:])
            lines.append(text.strip())

    return "\n".join(lines)


def normalize(text: str) -> str:
    text = text.lower()
    text = text.replace(" ","")
    text = text.replace("\n", "")
    return text


def evaluate_folder(folder_path: str):
    pipeline = ReceiptOCRPipeline()

    total_distance = 0
    total_chars = 0
    total_samples = 0


    start_time = time.time()

    for file in tqdm(image_files, desc="evaluating", unit="images"):

        image_path = os.path.join(folder_path, file)
        gt_path = os.path.join(folder_path, file.replace(".jpg", ".txt"))

        if not os.path.exist(gt_path):
            print(f"Missing GT for {file}")
            continue

        pred = pipeline.process_image(image_path)
        gt = load_ground_truth(gt_path)

        pred = normalize(pred)
        gt = normalize(gt)

        if len(gt) == 0:
            continue

        distance = editdistance.eval(pred, gt)

        total_distance += distance
        total_chars += len(gt)
        total_samples += 1

        # running cer
        running_cer = total_distance /total_chars

        if total_samples % 100 == 0:
            print(f"\nProcessed {total_samples} images")
            print(f"Current running CER: {running_cer:.4f}")


    total_time = time.time() - start_time
    dataset_cer = total_distance / total_chars

    print("\nFINAL RESULT")
    print(f"Total samples: {total_samples}")
    print(f"Dataset CER: {dataset_cer}")
    print(f"Dataset Accuracy (1-CER): {1 - dataset_cer:.4f}")
    print(f"Total time: {total_time}")

if __name__ == "__main__":
    evaluate_folder("")