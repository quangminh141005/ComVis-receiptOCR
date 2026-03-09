from paddleocr import PaddleOCR
import os

ocr = PaddleOCR(
    lang="en",
    det_limit_side_len=2000
)

img_path = r"C:\Users\admin\Documents\CV-receiptOCR\data\raw\X00016469670.jpg"

# thư mục lưu kết quả
output_dir = r"C:\Users\admin\Documents\CV-receiptOCR\data\results\txt"
os.makedirs(output_dir, exist_ok=True)

# lấy tên file ảnh
filename = os.path.splitext(os.path.basename(img_path))[0]

# tên file txt
output_txt = os.path.join(output_dir, filename + ".txt")

results = ocr.ocr(img_path)

with open(output_txt, "w", encoding="utf-8") as f:
    for line in results[0]:
        box = line[0]
        text = line[1][0].upper()

        coords = [str(int(p)) for point in box for p in point]
        line_out = ",".join(coords) + "," + text

        f.write(line_out + "\n")

print("Saved to:", output_txt)