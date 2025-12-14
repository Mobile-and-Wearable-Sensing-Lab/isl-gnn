import argparse
import math
import os
import random
import shutil

"""
Utility to split the ISL keypoints while ensuring that exactly one person's data is left out of the train/val sets
"""


def read_persons_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        persons = [line.strip() for line in f]

    return persons


def split_data(input_dir, root_dir, persons, person_selected_for_test, train_ratio=0.8235):
    random.seed(42)  # Ensure reproducibility

    output_dir = root_dir + "/dataset_split_" + person_selected_for_test
    os.makedirs(output_dir, exist_ok=True)

    # Validate selected person
    if person_selected_for_test not in persons:
        raise Exception(f"Invalid person selected to create test set. Person not available in existing list: {persons}")

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
        mov_files = [file for file in os.listdir(label_path) if file.endswith(".h5")]
        random.shuffle(mov_files)

        # Get test, train/val files
        person_label_for_file = [file.split(".")[0][-1] for file in mov_files]

        train_val_files = [mov_files[i]
                           for i in range(len(mov_files))
                           if person_label_for_file[i] != person_selected_for_test]
        test_files = [mov_files[i]
                      for i in range(len(mov_files))
                      if person_label_for_file[i] == person_selected_for_test]

        # Compute split indices for train/val
        total_files = len(train_val_files)
        train_end = math.ceil(total_files * train_ratio)

        # Split files
        train_files = train_val_files[:train_end]
        val_files = train_val_files[train_end:total_files]

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
                        help="The path the root directory where the dataset_split root directory is to be created.")
    parser.add_argument('--persons', type=str, help="The path the persons .txt file.")
    parser.add_argument('--person_selected_for_test', type=str, default="G",
                        help="The person selected to create the test set and leave out of the train/val set.")

    args = parser.parse_args()

    persons = read_persons_file(args.persons)

    if os.path.isdir(args.isldir):
        split_data(args.isldir, args.outdir, persons, args.person_selected_for_test)
    else:
        print("Invalid input/output directory path.")
