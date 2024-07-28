# import cv2
# import numpy as np
# import os
# from utils import mediapipe_detection, draw_styled_landmarks, extract_keypoints
# import mediapipe as mp
#
# # Set up paths and actions
# DATA_PATH = os.path.join('Dataset')
# actions = np.array(['hello', 'thank_you', 'i_love_you', 'yes', 'no'])
# no_sequences = 50
# sequence_length = 30
# start_folder = 0
#
# # Create folders for dataset
# for action in actions:
#     action_path = os.path.join(DATA_PATH, action)
#     if not os.path.exists(action_path):
#         os.makedirs(action_path)
#     dirmax = np.max(np.array(os.listdir(action_path)).astype(int)) if os.listdir(action_path) else 0
#     for sequence in range(1, no_sequences + 1):
#         try:
#             os.makedirs(os.path.join(action_path, str(dirmax + sequence)))
#         except FileExistsError:
#             pass
#
# def collect_data():
#     cap = cv2.VideoCapture(0)
#     exit_flag = False  # Flag to indicate when to exit
#
#     with mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
#         for action in actions:
#             if exit_flag:
#                 break
#             for sequence in range(start_folder, start_folder + no_sequences):
#                 if exit_flag:
#                     break
#                 for frame_num in range(sequence_length):
#                     ret, frame = cap.read()
#                     image, results = mediapipe_detection(frame, holistic)
#                     draw_styled_landmarks(image, results)
#
#                     if frame_num == 0:
#                         cv2.putText(image, 'STARTING COLLECTION', (120, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
#                         cv2.putText(image, f'Collecting frames for "{action}" Video Number "{sequence}"', (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
#                         cv2.imshow('Data Collection', image)
#                         cv2.waitKey(2000)
#                     else:
#                         cv2.putText(image, f'Collecting frames for "{action}" Video Number "{sequence}"', (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
#                         cv2.imshow('Data Collection', image)
#
#                     sequence_path = os.path.join(DATA_PATH, action, str(sequence))
#                     if not os.path.exists(sequence_path):
#                         os.makedirs(sequence_path)
#
#                     keypoints = extract_keypoints(results)
#                     npy_path = os.path.join(sequence_path, str(frame_num))
#                     np.save(npy_path, keypoints)
#
#                     if cv2.waitKey(10) & 0xFF == ord('q'):
#                         exit_flag = True
#                         break
#
#     cap.release()
#     cv2.destroyAllWindows()
#
# if __name__ == '__main__':
#     collect_data()


import cv2
import numpy as np
import os
from utils import mediapipe_detection, draw_styled_landmarks, extract_keypoints
import mediapipe as mp

# Set up paths and actions
DATA_PATH = os.path.join('Dataset')
actions = np.array(['hello', 'thank_you', 'i_love_you', 'yes', 'no'])
no_sequences = 50
sequence_length = 30
start_folder = 0

# Create folders for dataset
for action in actions:
    action_path = os.path.join(DATA_PATH, action)
    if not os.path.exists(action_path):
        os.makedirs(action_path)
    dirmax = np.max(np.array(os.listdir(action_path)).astype(int)) if os.listdir(action_path) else 0
    for sequence in range(1, no_sequences + 1):
        try:
            os.makedirs(os.path.join(action_path, str(dirmax + sequence)))
        except FileExistsError:
            pass

def collect_data():
    cap = cv2.VideoCapture(0)
    exit_flag = False  # Flag to indicate when to exit

    with mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for action in actions:
            if exit_flag:
                break
            for sequence in range(start_folder, start_folder + no_sequences):
                if exit_flag:
                    break
                for frame_num in range(sequence_length):
                    ret, frame = cap.read()
                    image, results = mediapipe_detection(frame, holistic)
                    draw_styled_landmarks(image, results)

                    if frame_num == 0:
                        cv2.putText(image, 'STARTING COLLECTION', (120, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                        cv2.putText(image, f'Collecting frames for "{action}" Video Number "{sequence}"', (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        cv2.imshow('Data Collection', image)
                        cv2.waitKey(2000)
                    else:
                        cv2.putText(image, f'Collecting frames for "{action}" Video Number "{sequence}"', (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        cv2.imshow('Data Collection', image)

                    sequence_path = os.path.join(DATA_PATH, action, str(sequence))
                    if not os.path.exists(sequence_path):
                        os.makedirs(sequence_path)

                    keypoints = extract_keypoints(results)
                    npy_path = os.path.join(sequence_path, str(frame_num))
                    np.save(npy_path, keypoints)

                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        exit_flag = True
                        break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    collect_data()
