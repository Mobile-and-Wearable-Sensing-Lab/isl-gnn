import argparse
import os
from pathlib import Path

"""
Utility to handle spaces in filenames of the ISL dataset.
Note - Spaces in the folder names were replaced with '_' manually as there were very few instances
"""


def clean_files(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(".mov") and "(" in file.lower():
                old_file_path = os.path.join(root, file)
                new_file_path = old_file_path[:-8] + "_" + old_file_path[-6] + ".MOV"

                old_name = Path(old_file_path)
                new_name = Path(new_file_path)
                old_name.rename(new_name)
                print(f"Renamed '{old_name}' to '{new_name}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Handle spaces in filenames of the ISL dataset.")

    parser.add_argument('--isldir', type=str, help="The root path to the ISL dataset.")

    args = parser.parse_args()

    if os.path.isdir(args.isldir):
        clean_files(args.isldir)
    else:
        print("Invalid directory path.")