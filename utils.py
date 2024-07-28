# import cv2
# import numpy as np
# import mediapipe as mp
#
# mp_holistic = mp.solutions.holistic
# mp_drawing = mp.solutions.drawing_utils
#
# def mediapipe_detection(image, model):
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     image.flags.writeable = False
#     results = model.process(image)
#     image.flags.writeable = True
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#     return image, results
#
# def draw_styled_landmarks(image, results, settings, visibility):
#     face_color, pose_color, left_hand_color, right_hand_color = settings
#     face_visible, pose_visible, left_hand_visible, right_hand_visible = visibility
#
#     if results.face_landmarks and face_visible:
#         mp_drawing.draw_landmarks(
#             image,
#             results.face_landmarks,
#             mp.solutions.face_mesh.FACEMESH_TESSELATION,
#             landmark_drawing_spec=None,
#             connection_drawing_spec=mp_drawing.DrawingSpec(color=face_color, thickness=1, circle_radius=1)
#         )
#     if results.pose_landmarks and pose_visible:
#         mp_drawing.draw_landmarks(
#             image,
#             results.pose_landmarks,
#             mp_holistic.POSE_CONNECTIONS,
#             landmark_drawing_spec=mp_drawing.DrawingSpec(color=pose_color, thickness=2, circle_radius=4),
#             connection_drawing_spec=mp_drawing.DrawingSpec(color=pose_color, thickness=2, circle_radius=2)
#         )
#     if results.left_hand_landmarks and left_hand_visible:
#         mp_drawing.draw_landmarks(
#             image,
#             results.left_hand_landmarks,
#             mp_holistic.HAND_CONNECTIONS,
#             landmark_drawing_spec=mp_drawing.DrawingSpec(color=left_hand_color, thickness=2, circle_radius=4),
#             connection_drawing_spec=mp_drawing.DrawingSpec(color=left_hand_color, thickness=2, circle_radius=2)
#         )
#     if results.right_hand_landmarks and right_hand_visible:
#         mp_drawing.draw_landmarks(
#             image,
#             results.right_hand_landmarks,
#             mp_holistic.HAND_CONNECTIONS,
#             landmark_drawing_spec=mp_drawing.DrawingSpec(color=right_hand_color, thickness=2, circle_radius=4),
#             connection_drawing_spec=mp_drawing.DrawingSpec(color=right_hand_color, thickness=2, circle_radius=2)
#         )
#
# def extract_keypoints(results):
#     try:
#         pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
#         face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
#         lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
#         rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
#         return np.concatenate([pose, face, lh, rh])
#     except Exception as e:
#         return None  # Return None to signal an error


import cv2
import numpy as np
import mediapipe as mp

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_styled_landmarks(image, results, settings=None, visibility=None):
    # Default colors
    default_settings = ((255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255))
    default_visibility = (True, True, True, True)

    if settings is None:
        settings = default_settings
    if visibility is None:
        visibility = default_visibility

    face_color, pose_color, left_hand_color, right_hand_color = settings
    face_visible, pose_visible, left_hand_visible, right_hand_visible = visibility

    if results.face_landmarks and face_visible:
        mp_drawing.draw_landmarks(
            image,
            results.face_landmarks,
            mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing.DrawingSpec(color=face_color, thickness=1, circle_radius=1)
        )
    if results.pose_landmarks and pose_visible:
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=pose_color, thickness=2, circle_radius=4),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=pose_color, thickness=2, circle_radius=2)
        )
    if results.left_hand_landmarks and left_hand_visible:
        mp_drawing.draw_landmarks(
            image,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=left_hand_color, thickness=2, circle_radius=4),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=left_hand_color, thickness=2, circle_radius=2)
        )
    if results.right_hand_landmarks and right_hand_visible:
        mp_drawing.draw_landmarks(
            image,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=right_hand_color, thickness=2, circle_radius=4),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=right_hand_color, thickness=2, circle_radius=2)
        )

def extract_keypoints(results):
    try:
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
        face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
        return np.concatenate([pose, face, lh, rh])
    except Exception as e:
        return None  # Return None to signal an error
