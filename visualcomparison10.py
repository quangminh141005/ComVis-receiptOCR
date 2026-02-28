"""
visualize_comparison.py
-----------------------
Opens each of the 10 test images side by side (original vs preprocessed).
Automatically reads the latest results from results.json (written by test10.py).

Usage:
    python visualize_comparison.py

Controls:
    SPACE or RIGHT arrow  -> next image
    LEFT  arrow           -> previous image
    Q or ESC              -> quit
"""

import json
import cv2
import numpy as np
from pathlib import Path

# -- The 10 test images ------------------------------------------------------
TEST_IMAGES = [
    "X00016469619.jpg",
    "X51008164510.jpg",
    "X51009453801.jpg",
    "X51008123604.jpg",
    "X51006619545.jpg",
    "X51005806679.jpg",
    "X51005675095.jpg",
    "X51005685357.jpg",
    "X51007339156.jpg",
    "X51005447861.jpg",
]

ORIGINAL_DIR     = "data/0325updated.task1train(626p)"
PREPROCESSED_DIR = "data/preprocessed_10test"
RESULTS_JSON     = "results.json"   # auto-written by test10.py
DISPLAY_HEIGHT   = 900


# ----------------------------------------------------------------------------
#  LOAD LATEST RESULTS FROM JSON
# ----------------------------------------------------------------------------
def load_results() -> dict:
    """Read results.json written by test10.py. Falls back to empty dict."""
    path = Path(RESULTS_JSON)
    if not path.exists():
        print(f"[WARN] {RESULTS_JSON} not found — run test10.py first.")
        return {}
    with open(path) as f:
        data = json.load(f)
    print(f"Loaded results from {RESULTS_JSON}")
    return data


# ----------------------------------------------------------------------------
#  IMAGE HELPERS
# ----------------------------------------------------------------------------
def load_and_resize(path: str, target_h: int) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        placeholder = np.ones((target_h, 400, 3), dtype=np.uint8) * 60
        cv2.putText(placeholder, "NOT FOUND", (20, target_h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 80, 255), 2)
        return placeholder
    h, w  = img.shape[:2]
    scale = target_h / h
    return cv2.resize(img, (int(w * scale), target_h))


def add_header(img: np.ndarray, text: str, bg_color: tuple) -> np.ndarray:
    header = np.zeros((50, img.shape[1], 3), dtype=np.uint8)
    header[:] = bg_color
    cv2.putText(header, text, (10, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    return np.vstack([header, img])


def build_comparison(fname: str, idx: int, total: int,
                     results: dict) -> np.ndarray:

    # Find original image path
    orig_path = None
    for p in Path(ORIGINAL_DIR).rglob(fname):
        orig_path = str(p)
        break
    pre_path = str(Path(PREPROCESSED_DIR) / fname)

    orig_img = load_and_resize(orig_path or "", DISPLAY_HEIGHT)
    pre_img  = load_and_resize(pre_path,        DISPLAY_HEIGHT)

    # Ensure both are BGR for display
    if len(orig_img.shape) == 2:
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_GRAY2BGR)
    if len(pre_img.shape) == 2:
        pre_img  = cv2.cvtColor(pre_img,  cv2.COLOR_GRAY2BGR)

    # Pull metrics from latest results
    r        = results.get(fname, {})
    orig_cer = r.get("orig_cer", 0.0)
    pre_cer  = r.get("pre_cer",  0.0)
    orig_wer = r.get("orig_wer", 0.0)
    pre_wer  = r.get("pre_wer",  0.0)
    status   = r.get("status",   "UNKNOWN")

    delta_cer    = orig_cer - pre_cer
    delta_wer    = orig_wer - pre_wer
    cer_arrow    = "down" if delta_cer > 0 else "up"
    wer_arrow    = "down" if delta_wer > 0 else "up"

    orig_color   = (60, 60, 60)
    pre_color    = (0, 140, 0) if status == "IMPROVED" else \
                   (0, 0, 200) if status == "WORSE"    else (120, 120, 0)
    status_label = status if r else "NO RESULTS — run test10.py"

    orig_img = add_header(
        orig_img,
        f"ORIGINAL   CER:{orig_cer:.3f}  WER:{orig_wer:.3f}",
        orig_color,
    )
    pre_img = add_header(
        pre_img,
        f"PREPROCESSED   CER:{pre_cer:.3f}({cer_arrow} {abs(delta_cer):.3f})"
        f"  WER:{pre_wer:.3f}({wer_arrow} {abs(delta_wer):.3f})  [{status_label}]",
        pre_color,
    )

    # Pad both panels to same width before hstack
    h1, w1 = orig_img.shape[:2]
    h2, w2 = pre_img.shape[:2]
    max_w  = max(w1, w2)
    if w1 < max_w:
        orig_img = cv2.copyMakeBorder(orig_img, 0, 0, 0, max_w - w1,
                                      cv2.BORDER_CONSTANT, value=(40,40,40))
    if w2 < max_w:
        pre_img  = cv2.copyMakeBorder(pre_img,  0, 0, 0, max_w - w2,
                                      cv2.BORDER_CONSTANT, value=(40,40,40))

    combined = np.hstack([orig_img, pre_img])

    # Bottom nav bar
    nav_bar = np.zeros((45, combined.shape[1], 3), dtype=np.uint8)
    nav_bar[:] = (25, 25, 25)
    nav_text = (f"  [{idx+1}/{total}]  {fname}  "
                f"|  SPACE / -> : next   <- : prev   Q / ESC : quit")
    cv2.putText(nav_bar, nav_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
    return np.vstack([combined, nav_bar])


# ----------------------------------------------------------------------------
#  MAIN
# ----------------------------------------------------------------------------
def main():
    results = load_results()

    print("\nOpening visual comparison window...")
    print("Controls: SPACE/-> = next  |  <- = prev  |  Q/ESC = quit\n")

    idx   = 0
    total = len(TEST_IMAGES)

    while True:
        fname = TEST_IMAGES[idx]
        frame = build_comparison(fname, idx, total, results)

        # Fit to screen width if too wide
        screen_w = 1800
        fh, fw   = frame.shape[:2]
        if fw > screen_w:
            scale = screen_w / fw
            frame = cv2.resize(frame, (screen_w, int(fh * scale)))

        cv2.imshow("Receipt Preprocessing Comparison  (latest results)", frame)
        key = cv2.waitKey(0) & 0xFF

        if key in [ord('q'), 27]:        # Q / ESC -> quit
            break
        elif key in [ord(' '), 83, 100]: # SPACE / -> / d -> next
            idx = (idx + 1) % total
        elif key in [81, 97]:            # <- / a -> previous
            idx = (idx - 1) % total

    cv2.destroyAllWindows()
    print("Closed.")


if __name__ == "__main__":
    main()