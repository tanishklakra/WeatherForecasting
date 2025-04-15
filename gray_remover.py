import os
import argparse
from PIL import Image
import shutil
import cv2

def is_grayscale(img_path):
    """
    Check if an image is grayscale (has only one channel or all RGB channels are equal).
    """
    try:
        with Image.open(img_path) as img:
            # Convert image to RGB mode if it's not already
            if img.mode != 'RGB':
                rgb_img = img.convert('RGB')
            else:
                rgb_img = img

            # Take a sample of pixels to check if the image is grayscale
            pixels = list(rgb_img.getdata())

            # Check first 100 pixels (or all pixels if less than 100)
            sample_size = min(100, len(pixels))
            for i in range(sample_size):
                r, g, b = pixels[i]
                # If any pixel has different RGB values, the image is not grayscale
                if r != g or g != b:
                    return False

            return True
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return False


def process_dataset(dataset_path, backup_folder=None, dry_run=False):
    """
    Process the dataset structure and remove grayscale images.

    Structure: dataset → train/val → rainy/cloudy/sunshine/sunrise → images

    Args:
        dataset_path: Path to the dataset folder
        backup_folder: Optional path to back up removed images
        dry_run: If True, only print actions without removing files
    """
    # Create backup folder if specified
    if backup_folder and not dry_run:
        os.makedirs(backup_folder, exist_ok=True)

    total_removed = 0
    total_processed = 0

    # Valid image extensions
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif', '.webp']

    # Process the dataset structure
    for split in ['train', 'val']:
        split_path = os.path.join(dataset_path, split)
        if not os.path.isdir(split_path):
            print(f"Warning: {split_path} not found, skipping")
            continue

        for category in ['rainy', 'cloudy', 'sunshine', 'sunrise']:
            category_path = os.path.join(split_path, category)
            if not os.path.isdir(category_path):
                print(f"Warning: {category_path} not found, skipping")
                continue

            print(f"\nProcessing {split}/{category}:")
            category_processed = 0
            category_removed = 0

            # Process all files in the category folder
            for filename in os.listdir(category_path):
                file_path = os.path.join(category_path, filename)

                # Skip directories and non-image files
                if os.path.isdir(file_path):
                    continue

                file_ext = os.path.splitext(filename)[1].lower()
                if file_ext not in valid_extensions:
                    continue

                category_processed += 1

                # Check if the image is grayscale
                if is_grayscale(file_path):
                    category_removed += 1
                    if dry_run:
                        print(f"  Would remove: {filename}")
                    else:
                        if backup_folder:
                            # Create nested backup directory structure
                            backup_category_path = os.path.join(backup_folder, split, category)
                            os.makedirs(backup_category_path, exist_ok=True)

                            backup_path = os.path.join(backup_category_path, filename)
                            print(f"  Moving grayscale image to backup: {filename}")
                            shutil.move(file_path, backup_path)
                        else:
                            print(f"  Removing grayscale image: {filename}")
                            os.remove(file_path)

            print(
                f"  Summary for {split}/{category}: Processed {category_processed} images, found {category_removed} grayscale images")
            total_processed += category_processed
            total_removed += category_removed

    print("\n" + "=" * 50)
    print(f"DATASET SUMMARY: Processed {total_processed} images")
    print(f"Found {total_removed} grayscale images")

    if dry_run:
        print("No files were removed (dry run)")
    else:
        print(f"Removed/moved {total_removed} grayscale images")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove grayscale images from dataset structure")
    parser.add_argument("dataset", help="Path to the dataset folder")
    parser.add_argument("--backup", help="Path to backup folder for removed images (optional)")
    parser.add_argument("--dry-run", action="store_true", help="Only print what would be removed, without deleting")

    args = parser.parse_args()

    if not os.path.isdir(args.dataset):
        print(f"Error: {args.dataset} is not a valid directory")
        exit(1)

    process_dataset(args.dataset, args.backup, args.dry_run)