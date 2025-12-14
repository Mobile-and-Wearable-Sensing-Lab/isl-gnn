import mediapipe as mp
import cv2
import numpy as np

from cv2_utils import file_iters, load_file


"""
Contains utility functions to generate, clean, denormalize and visualize the hand landmarks using Mediapipe
"""


def generate_hand_landmarks(hand_detection_module, num_hands_for_sign, input_file):
    if num_hands_for_sign == 1:
        return generate_one_handed_sign_landmarks(hand_detection_module, input_file)
    elif num_hands_for_sign == 2:
        return generate_two_handed_sign_landmarks(hand_detection_module, input_file)
    else:
        raise Exception(f"Invalid number of hands: {num_hands_for_sign}")


def generate_one_handed_sign_landmarks(hand_detection_module, input_file):
    results = np.zeros((file_iters(input_file), 1, 21, 2))

    for idx, frame in enumerate(load_file(input_file)):
        hand_landmarks = run_hand_detection(hand_detection_module, frame)
        results[idx, :, :, :] = process_one_handed_sign_landmarks_post_detection(hand_landmarks, coords="xy")

    return results


def generate_two_handed_sign_landmarks(hand_detection_module, input_file):
    results = np.zeros((file_iters(input_file), 2, 21, 2))

    for idx, frame in enumerate(load_file(input_file)):
        hand_landmarks = run_hand_detection(hand_detection_module, frame)
        results[idx, :, :, :] = process_two_handed_sign_landmarks_post_detection(hand_landmarks, coords="xy")

    return results


def run_hand_detection(hand_detection_module: mp.solutions.hands.Hands, frame, colorspace_convert=cv2.COLOR_BGR2RGB, **kwargs):
    if 'flipH' in kwargs and kwargs['flipH']:
        cv2.flip(frame, 1)
    if 'flipV' in kwargs and kwargs['flipV']:
        cv2.flip(frame, 0)

    if colorspace_convert:
        frame = cv2.cvtColor(frame, colorspace_convert)
    return hand_detection_module.process(frame)


# TODO - TO BE DEPRECATED
def process_one_handed_sign_landmarks_post_detection_old(results, keypoints=mp.solutions.hands.HandLandmark, coords="xyz"):
    lm_list = results.multi_hand_landmarks

    if lm_list:
        return np.array([[[
            1 - getattr(lm.landmark[keypoint], coord)
            if results.multi_handedness[idx].classification[0].label.lower() == "left" and coord == "x"
            else getattr(lm.landmark[keypoint], coord)
            for coord in coords
        ] for keypoint in keypoints] for idx, lm in enumerate(lm_list)])
    else:
        return np.zeros((1, 21, len(coords)))


def process_one_handed_sign_landmarks_post_detection(results, keypoints=mp.solutions.hands.HandLandmark, coords="xyz"):
    lm_list = results.multi_hand_landmarks

    if lm_list:
        final_landmarks = np.array([
            [
                [getattr(lm.landmark[keypoint], coord) for coord in coords]
                for keypoint in keypoints
            ]
            for idx, lm in enumerate(lm_list)
            if results.multi_handedness[idx].classification[0].label.lower() == "left" and results.multi_handedness[idx].classification[0].score > 0.95
        ])

        # Ensure that the output shape is of the form (1, 21, len(coords))
        if len(final_landmarks.shape) != 3:
            return np.zeros((1, 21, len(coords)))
        elif final_landmarks.shape[0] != 1:
            return [final_landmarks[0]]

        return final_landmarks
    else:
        return np.zeros((1, 21, len(coords)))


def process_two_handed_sign_landmarks_post_detection(results, keypoints=mp.solutions.hands.HandLandmark, coords="xyz"):
    lm_list = results.multi_hand_landmarks

    if lm_list:
        max_handedness_confidence = dict({"left": 0, "right": 0})

        for idx, lm in enumerate(lm_list):
            handedness = results.multi_handedness[idx].classification[0].label.lower()
            max_handedness_confidence[handedness] = max(
                max_handedness_confidence[handedness],
                results.multi_handedness[idx].classification[0].score
            )

        final_landmarks = np.array([
            [
                [getattr(lm.landmark[keypoint], coord) for coord in coords]
                for keypoint in keypoints
            ]
            for idx, lm in enumerate(lm_list)
            if results.multi_handedness[idx].classification[0].score >= max_handedness_confidence[results.multi_handedness[idx].classification[0].label.lower()]
        ])

        # Ensure that the output shape is of the form (2, 21, len(coords))
        if len(final_landmarks.shape) != 3:
            return np.zeros((2, 21, len(coords)))

        elif final_landmarks.shape[0] != 2:
            padded_final_landmarks = np.zeros((2, 21, len(coords)))
            single_detected_hand_handedness = None

            if results.multi_handedness[0].classification[0].score > 0.95:
                single_detected_hand_handedness = results.multi_handedness[0].classification[0].label.lower()

            if single_detected_hand_handedness == "left":
                padded_final_landmarks[:1, :, :] = final_landmarks

            elif single_detected_hand_handedness == "right":
                padded_final_landmarks[1:, :, :] = final_landmarks

            return padded_final_landmarks

        return final_landmarks
    else:
        return np.zeros((2, 21, len(coords)))


def clean_hand_landmarks(landmarks):
    mask = ~np.all(landmarks == 0, axis=(1, 2, 3))  # Keep only non-zero 3D arrays
    filtered_array = landmarks[mask]
    return filtered_array


def denormalize_hand_landmarks(landmarks, frame_width, frame_height):
    return (landmarks * np.array([frame_width, frame_height])).astype(int)


def create_and_save_hand_landmarks_video(landmarks, frame_width, frame_height, output_file, fps=30):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

    for landmark in landmarks:
        frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)  # Blank frame

        for hands in landmark:  # Loop over hands
            for x, y in hands:
                cv2.circle(frame, (x, y), radius=5, color=(0, 255, 0), thickness=-1)  # Green keypoints

        out.write(frame)

    out.release()
    print(f"Hand landmarks video saved as {output_file}")


