import argparse
import math
import os
import random
import shutil

"""
Utility to split the ISL keypoints while ensuring that the distribution of number of samples across classes remains the in train/val/test
"""
def split_data(input_dir, output_dir, train_ratio=0.7, test_ratio=0.15):
    random.seed(42)  # Ensure reproducibility

    val_ratio = 1 - train_ratio - test_ratio

    # Define output paths
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    test_dir = os.path.join(output_dir, "test")

    # Create train, val, test directories
    for split in [train_dir, val_dir, test_dir]:
        os.makedirs(split, exist_ok=True)

    # Iterate through label directories
    for label in os.listdir(input_dir):
        label_path = "\ ".join(os.path.join(input_dir, label).split(" "))
        if not os.path.isdir(label_path):
            continue

        # Get all .h5 files
        mov_files = [f for f in os.listdir(label_path) if f.endswith(".h5")]
        random.shuffle(mov_files)

        # Compute split indices
        total_files = len(mov_files)
        train_end = math.ceil(total_files * train_ratio)
        val_end = train_end + math.ceil(total_files * val_ratio)

        # Split files
        train_files = mov_files[:train_end]
        val_files = mov_files[train_end:val_end]
        test_files = mov_files[val_end:]

        # Copy files to respective directories
        for split, files in zip([train_dir, val_dir, test_dir], [train_files, val_files, test_files]):
            split_label_dir = os.path.join(split, label)
            os.makedirs(split_label_dir, exist_ok=True)
            for file in files:
                shutil.copy(os.path.join(label_path, file), os.path.join(split_label_dir, file))

    print("Data split completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split the ISL keypoints while ensuring that the distribution of number of samples across classes remains the in train/val/test.")

    parser.add_argument('--isldir', type=str, help="The root path to the ISL extracted keypoints.")
    parser.add_argument('--outdir', type=str,
                        help="The path the root directory where the train/val/test directories are to be created.")
    parser.add_argument('--train_ratio', type=float, help="The ratio of samples to be considered for creating the train split.")
    parser.add_argument('--test_ratio', type=float, help="The ratio of samples to be considered for creating the test split.")

    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    if os.path.isdir(args.isldir):
        split_data(args.isldir, args.outdir, args.train_ratio, args.test_ratio)
    else:
        print("Invalid input/output directory path.")
