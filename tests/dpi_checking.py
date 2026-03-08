import os
import glob
from PIL import Image

def check_dpi_range(dataset_path):
    print(f"Scanning directory: {dataset_path}...\n")
    
    # Check for common receipt image formats
    extensions = ('*.jpg', '*.jpeg', '*.png')
    image_files = []
    
    # Grab all images, including those in subdirectories
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(dataset_path, '**', ext), recursive=True))

    if not image_files:
        print("No images found in the specified path. Please check your directory.")
        return

    dpis_x = []
    dpis_y = []
    missing_dpi_count = 0

    # Read metadata for each image
    for img_path in image_files:
        try:
            with Image.open(img_path) as img:
                # Pillow stores DPI as a tuple in the info dictionary (dpi_x, dpi_y)
                dpi = img.info.get('dpi')
                if dpi:
                    dpis_x.append(dpi[0])
                    dpis_y.append(dpi[1])
                else:
                    missing_dpi_count += 1
        except Exception as e:
            print(f"Error reading {img_path}: {e}")

    # Output the results
    print("-" * 30)
    print(f"Total images checked: {len(image_files)}")
    print(f"Images with NO DPI metadata: {missing_dpi_count}")
    
    if dpis_x and dpis_y:
        print(f"X-axis DPI Range: {min(dpis_x)} to {max(dpis_x)}")
        print(f"Y-axis DPI Range: {min(dpis_y)} to {max(dpis_y)}")
        
        # Optional: Print the average DPI
        avg_x = sum(dpis_x) / len(dpis_x)
        avg_y = sum(dpis_y) / len(dpis_y)
        print(f"Average DPI: ~{round(avg_x)}x{round(avg_y)}")
    else:
        print("Could not find DPI metadata in any of the images.")

dataset_path = 'data/SROIE2019/0325updated.task1train(626p)' 
check_dpi_range(dataset_path)