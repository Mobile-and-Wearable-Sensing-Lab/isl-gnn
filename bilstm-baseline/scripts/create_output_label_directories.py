import argparse
from pathlib import Path


def read_labels_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f]

    return labels


def create_label_directories(root_directory_path, labels):
    for label in labels:
        folder_path = Path(root_directory_path + "/" + label)
        folder_path.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create output directories for the specified labels.")

    parser.add_argument('--labels', type=str, help="The path the labels .txt file.")
    parser.add_argument('--outdir', type=str, help="The path the root directory where the label directories are to be created.")

    args = parser.parse_args()

    labels = read_labels_file(args.labels)
    create_label_directories(args.outdir, labels)
