import argparse

from h5py_utils import write_file
from hand_landmarks_utils import *

"""
Utility to generate and save hand landmarks from input video/image using Mediapipe
"""


def main():
    parser = argparse.ArgumentParser(
        description="Generate and save the hand landmarks from input video/image using Mediapipe.")

    parser.add_argument('--input_file', type=str, help="Path to the input video file.")
    parser.add_argument('--output_file', type=str, help="Path to the output h5py file.")
    parser.add_argument('--num_hands_for_sign', type=int, default=1,
                        help="Number of hands used for making the signs that are to be detected.")

    parser.add_argument('--static_image_mode', type=bool, default=False,
                        help="The STATIC_IMAGE_MODE property to be set in Mediapipe hands module.")

    parser.add_argument('--max_num_hands', type=int, default=2,
                        help="The MAX_NUM_HANDS property to be set in Mediapipe hands module.")

    parser.add_argument('--min_detection_confidence', type=float, default=0.5,
                        help="The MIN_DETECTION_CONFIDENCE property to be set in Mediapipe hands module.")

    parser.add_argument('--min_tracking_confidence', type=float, default=0.5,
                        help="The MIN_TRACKING_CONFIDENCE property to be set in Mediapipe hands module.")

    args = parser.parse_args()

    print(f"STATIC_IMAGE_MODE: {args.static_image_mode}")
    print(f"MAX_NUM_HANDS: {args.max_num_hands}")
    print(f"MIN_DETECTION_CONFIDENCE: {args.min_detection_confidence}")
    print(f"MIN_TRACKING_CONFIDENCE: {args.min_tracking_confidence}")

    hand_detection_module = mp.solutions.hands.Hands(
        static_image_mode=args.static_image_mode,
        max_num_hands=args.max_num_hands,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence
    )

    input_file = f"/in/{args.input_file}"
    output_file = f"/out/{args.output_file}"

    landmarks = generate_hand_landmarks(hand_detection_module, args.num_hands_for_sign, input_file)
    # cleaned_landmarks = clean_hand_landmarks(landmarks)
    write_file(output_file, landmarks)


if __name__ == "__main__":
    main()

