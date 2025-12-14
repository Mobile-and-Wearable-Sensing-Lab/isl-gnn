import argparse
import os

import cv2
import matplotlib.pyplot as plt

from utils import read_labels_file

"""
Utility to visualize the distribution of frame counts in the ISL dataset
"""


def get_frame_count(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open {video_path}")
        return None
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count


def find_mov_files(directory, labels):
    mov_files = []
    for root, _, files in os.walk(directory):
        label = root.split("/")[-1].split(".")[-1].strip(" ")
        if label not in labels:
            continue

        for file in files:
            if file.lower().endswith(".mov"):
                mov_files.append(os.path.join(root, file))
    return mov_files


def plot_distribution(frame_counts):
    plt.figure(figsize=(10, 6))
    plt.hist(frame_counts.values(), bins=20, edgecolor='black', alpha=0.7)
    plt.xlabel("Number of Frames")
    plt.ylabel("Frequency")
    plt.title("Distribution of Frame Counts in .mov Files")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize the distribution of frame counts in the ISL dataset.")

    parser.add_argument('--labels', type=str, help="The path the labels .txt file.")
    parser.add_argument('--isldir', type=str, help="The root path to the ISL dataset.")

    args = parser.parse_args()

    if not os.path.isdir(args.isldir):
        print("Invalid directory path.")
        return

    labels = read_labels_file(args.labels)
    mov_files = find_mov_files(args.isldir, labels)

    if not mov_files:
        print("No .mov files found.")
        return

    frame_counts = {}
    total_frames = 0

    for file in mov_files:
        frame_count = get_frame_count(file)
        if frame_count is not None:
            frame_counts[file] = frame_count
            total_frames += frame_count

    for file, count in frame_counts.items():
        print(f"{file}: {count} frames")

    if frame_counts:
        average_frames = total_frames / len(frame_counts)
        print(f"Average number of frames: {average_frames:.2f}")
        plot_distribution(frame_counts)


if __name__ == "__main__":
    main()
