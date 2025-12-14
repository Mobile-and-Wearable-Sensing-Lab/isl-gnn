import argparse

from cv2_utils import get_video_frame_dimensions
from hand_landmarks_utils import *

"""
Utility to visualize the hand landmarks generated using Mediapipe as a video
"""


def main():
    parser = argparse.ArgumentParser(description="Visualize the hand landmarks generated using Mediapipe as a video.")

    parser.add_argument('--input_file', type=str, help="Path to the input video file.")
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

    hand_detection_module = mp.solutions.hands.Hands(
        static_image_mode=args.static_image_mode,
        max_num_hands=args.max_num_hands,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence
    )

    print(f"STATIC_IMAGE_MODE: {args.static_image_mode}")
    print(f"MAX_NUM_HANDS: {args.max_num_hands}")
    print(f"MIN_DETECTION_CONFIDENCE: {args.min_detection_confidence}")
    print(f"MIN_TRACKING_CONFIDENCE: {args.min_tracking_confidence}")

    input_file = f"/in/{args.input_file}"

    input_file_split = args.input_file.split("/")
    output_file = f"/out/hand_landmarks_viz_{input_file_split[0]}_{input_file_split[1].split(' ')[1]}_{input_file_split[2][:-4]}.mp4"

    frame_width, frame_height = get_video_frame_dimensions(input_file)

    landmarks = generate_hand_landmarks(hand_detection_module, args.num_hands_for_sign, input_file)
    # cleaned_landmarks = clean_hand_landmarks(landmarks)
    denormalized_landmarks = denormalize_hand_landmarks(landmarks, frame_width, frame_height)

    create_and_save_hand_landmarks_video(denormalized_landmarks, frame_width, frame_height, output_file)


if __name__ == "__main__":
    main()

