import argparse
import os

from utils import read_labels_file


def generate_transformer_commands(directory, labels, num_hands_for_sign, output_script):
    with open(output_script, "w", encoding="utf-8") as f:
        f.write("#!/bin/bash\n\n")
        f.write("counter=0\n\n")

        for root, _, files in os.walk(directory):
            label = root.split("/")[-1].split(".")[-1].strip(" ")
            if label not in labels:
                continue

            for file in files:
                if file.lower().endswith(".mov"):
                    temp1 = "\ ".join(os.path.join(root, file).split(" "))
                    prefix = directory if directory[-1] != "/" else f"{directory}/"
                    relative_input_path = temp1.removeprefix(prefix)

                    output_file = file.replace(".MOV", ".h5")
                    relative_output_path = f"{label}/{output_file}"
                    relative_output_path = "\ ".join(relative_output_path.split(" "))

                    f.write(f"python3 generate_hand_landmarks.py --input_file {relative_input_path} --output_file {relative_output_path} --num_hands_for_sign {num_hands_for_sign}\n")
                    f.write("counter=$((counter + 1))\n")
                    f.write('echo "Counter: $counter"\n\n')

    print(f"Script written to {output_script}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create shell commands to generate ISL keypoints data.")

    parser.add_argument('--isldir', type=str, help="The root path to the ISL dataset.")
    parser.add_argument('--labels', type=str, help="The path the labels .txt file.")
    parser.add_argument('--num_hands_for_sign', type=int, default=1, help="The path the labels .txt file.")
    parser.add_argument('--shell_script_path', type=str, help="The path where the shell script is to be created.")

    args = parser.parse_args()

    if os.path.isdir(args.isldir):
        labels = read_labels_file(args.labels)
        generate_transformer_commands(args.isldir, labels, args.num_hands_for_sign, args.shell_script_path)
    else:
        print("Invalid directory path.")
