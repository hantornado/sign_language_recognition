

# # Initialize Pygame for voice output
# pygame.mixer.init()
# print("Pygame initialized for audio playback.")
#
# # Set page configuration at the top
# st.set_page_config(page_title="Sign Language Recognition", layout="wide", initial_sidebar_state="expanded")
#
# # Initialize the translator
# translator = Translator()
#
# # Ensure audio folder exists in assets
# audio_folder = os.path.join('assets', 'audio_files')
# if not os.path.exists(audio_folder):
#     os.makedirs(audio_folder)
#
# # Ensure fonts folder exists
# fonts_folder = os.path.join('assets', 'fonts')
# if not os.path.exists(fonts_folder):
#     raise FileNotFoundError(f"Fonts folder not found: {fonts_folder}")

# def speak_word(text, lang_code):
#     tts = gTTS(text, lang=lang_code)
#     filename = f"voice_{int(time.time())}.mp3"  # Unique filename using current timestamp
#     tts.save(filename)
#     sound = pygame.mixer.Sound(filename)
#     sound.play()
#
#     # Ensure the file is deleted after playing
#     def cleanup():
#         time.sleep(sound.get_length())
#         os.remove(filename)
#
#     cleanup_thread = threading.Thread(target=cleanup)
#     cleanup_thread.start()

import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import pygame
import threading
from googletrans import Translator
import time
from collections import deque
from utils import mediapipe_detection, draw_styled_landmarks, extract_keypoints
import os
import mediapipe as mp
from concurrent.futures import ThreadPoolExecutor

# Initialize Pygame for voice output
pygame.mixer.init()
print("Pygame initialized for audio playback.")

# Set page configuration at the top
st.set_page_config(page_title="Sign Language Recognition", layout="wide", initial_sidebar_state="expanded")

# Initialize the translator
translator = Translator()

# Ensure audio folder exists in assets
audio_folder = os.path.join('assets', 'audio_files')
if not os.path.exists(audio_folder):
    os.makedirs(audio_folder)

# Ensure fonts folder exists
fonts_folder = os.path.join('assets', 'fonts')
if not os.path.exists(fonts_folder):
    raise FileNotFoundError(f"Fonts folder not found: {fonts_folder}")

# Ensure CSS file exists
css_path = os.path.join('assets', 'style.css')
if not os.path.exists(css_path):
    raise FileNotFoundError(f"CSS file not found: {css_path}")

# Load custom CSS
with open(css_path) as css:
    st.markdown(f'<style>{css.read()}</style>', unsafe_allow_html=True)

# Function Definitions
def speak_word(word, lang_code):
    try:
        file_path = os.path.join(audio_folder, f"{word}_{lang_code}.mp3")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            continue
    except Exception as e:
        print(f"Failed to play audio! Error: {e}")

def translate_text(word, lang_code):
    try:
        translation = translator.translate(word, dest=lang_code)
        return translation.text if translation and translation.text else word
    except Exception as e:
        print(f"Translation error: {e}")
        return word

def update_history(history, word, lang_code):
    if lang_code != "en":
        translated_word = translate_text(word, lang_code)
    else:
        translated_word = word
    history.insert(0, translated_word.replace("_", " "))
    if len(history) > 10:
        history.pop()

# Load the trained model
model_path = os.path.join('models', 'best_model.h5')
model = tf.keras.models.load_model(model_path)

# Actions and label map
actions = np.array(['hello', 'thank_you', 'i_love_you', 'yes', 'no'])
label_map = {num: label for num, label in enumerate(actions)}

# Enhanced prediction logic
sequence = deque(maxlen=30)  # Use deque for efficient fixed-length sequence
sentence = []
predictions = deque(maxlen=10)  # Smoothing over last 10 predictions
threshold = 0.7  # Increased threshold for higher confidence

# Initialize Mediapipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Initialize variables for real-time prediction
sequence = []
sentence = []
predictions = []
threshold = 0.5
max_idx = None
speak_thread = None
last_recognized_signs = set()
translation_result = ""
executor = ThreadPoolExecutor(max_workers=1)
recognition_timeout = 5  # seconds
last_recognized_time = time.time()

# Initialize session state for settings
if 'translate_to' not in st.session_state:
    st.session_state.translate_to = "English"
if 'voice_output' not in st.session_state:
    st.session_state.voice_output = True
if 'is_running' not in st.session_state:
    st.session_state.is_running = False
if 'stop_event' not in st.session_state:
    st.session_state.stop_event = threading.Event()
if 'history' not in st.session_state:
    st.session_state.history = []
if 'last_output_time' not in st.session_state:
    st.session_state.last_output_time = 0

# Language mapping
language_map = {
    "English": "en",
    "Chinese": "zh-cn",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Japanese": "ja"
}

# Sidebar for settings
with st.sidebar:
    st.title("Settings")
    st.header("Landmark visibility")
    face_landmark_visibility = st.checkbox("Face Landmark", True)
    pose_landmark_visibility = st.checkbox("Pose Landmark", True)
    left_hand_landmark_visibility = st.checkbox("Left Hand Landmark", True)
    right_hand_landmark_visibility = st.checkbox("Right Hand Landmark", True)

    st.header("Landmark colours")
    face_landmark_colour = st.color_picker("Face Landmark Colour", "#E0E0E0")
    pose_landmark_colour = st.color_picker("Pose Landmark Colour", "#66CC00")
    left_hand_landmark_colour = st.color_picker("Left Hand Landmark Colour", "#6666FF")
    right_hand_landmark_colour = st.color_picker("Right Hand Landmark Colour", "#66B2FF")

    st.header("Translation settings")
    st.session_state.translate_to = st.selectbox("Translate to", list(language_map.keys()))
    st.session_state.voice_output = st.checkbox("Enable voice output", True)

# Main page layout
st.markdown("<h1>Sign Language Recognition</h1>", unsafe_allow_html=True)

# Create layout with columns
col1, col2 = st.columns([2, 1])

with col1:
    camera_window = st.empty()
    confidence_placeholder = st.empty()

    if st.session_state.is_running:
        if st.button("Stop"):
            st.session_state.is_running = False
            st.session_state.stop_event.set()
            st.experimental_rerun()
    else:
        if st.button("Start"):
            st.session_state.is_running = True
            st.session_state.stop_event.clear()
            st.experimental_rerun()

with col2:
    st.markdown("<div id='history-section'><h2>History</h2></div>", unsafe_allow_html=True)
    history_placeholder = st.empty()

def hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip('#')
    rgb = tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return (rgb[2], rgb[1], rgb[0])  # Convert to BGR

if st.session_state.is_running:
    cap = cv2.VideoCapture(0)
    last_prediction_time = time.time()
    prediction_interval = 1  # seconds between predictions

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened() and not st.session_state.stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break

            if 'last_prediction_time' not in st.session_state:
                st.session_state.last_prediction_time = time.time()

            # Make detections
            image, results = mediapipe_detection(frame, holistic)
            landmark_settings = (
                hex_to_bgr(face_landmark_colour),
                hex_to_bgr(pose_landmark_colour),
                hex_to_bgr(left_hand_landmark_colour),
                hex_to_bgr(right_hand_landmark_colour)
            )

            visibility_settings = (
                face_landmark_visibility,
                pose_landmark_visibility,
                left_hand_landmark_visibility,
                right_hand_landmark_visibility
            )

            # Draw landmarks based on settings
            draw_styled_landmarks(image, results, landmark_settings, visibility_settings)

            # Extract keypoints
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]

            if len(sequence) == 30 and (time.time() - st.session_state.last_prediction_time) > prediction_interval:
                # Make a prediction with the model
                res = model.predict(np.expand_dims(sequence, axis=0))[0]

                predictions.append(res)
                max_idx = np.argmax(res)

                word = actions[max_idx]
                lang_code = language_map[st.session_state.translate_to]

                # Ensure the same sign is not recognized repeatedly within the timeout period
                current_time = time.time()
                if (current_time - last_recognized_time) > recognition_timeout and (current_time - st.session_state.last_output_time) > recognition_timeout:
                    if st.session_state.voice_output:
                        # Speak the word in the translated language
                        thread = threading.Thread(target=speak_word, args=(word, lang_code))
                        thread.start()

                    print(word)
                    word = word.replace("_", " ")
                    update_history(st.session_state.history, word, lang_code)
                    st.session_state.last_prediction_time = current_time
                    st.session_state.last_output_time = current_time
                    last_recognized_signs = word
                    last_recognized_time = current_time

            if max_idx is not None:
                confidence_placeholder.markdown(
                    f"<h3 style='color: red;'>Detected action: {actions[max_idx].replace('_', ' ')}</h3>",
                    unsafe_allow_html=True)

            # Display the frame in the Streamlit app
            camera_window.image(image, channels='BGR')

            # Display the history of recognized signs
            history_text = "\n".join(st.session_state.history)
            history_placeholder.markdown(f"<div class='history-text'>{history_text}</div>", unsafe_allow_html=True)
    predictions.clear()

    cap.release()
    cv2.destroyAllWindows()













# if st.session_state.is_running:
#     cap = cv2.VideoCapture(0)
#     last_prediction_time = time.time()
#     prediction_interval = 1  # seconds between predictions
#
#     with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
#         while cap.isOpened() and not st.session_state.stop_event.is_set():
#             ret, frame = cap.read()
#             if not ret:
#                 break
#
#             if 'last_prediction_time' not in st.session_state:
#                 st.session_state.last_prediction_time = time.time()
#
#             # Make detections
#             image, results = mediapipe_detection(frame, holistic)
#             landmark_settings = (
#                 hex_to_rgb(face_landmark_colour),
#                 hex_to_rgb(pose_landmark_colour),
#                 hex_to_rgb(left_hand_landmark_colour),
#                 hex_to_rgb(right_hand_landmark_colour)
#             )
#             draw_styled_landmarks(image, results, landmark_settings)
#
#             # Extract keypoints
#             keypoints = extract_keypoints(results)
#             sequence.append(keypoints)
#             sequence = sequence[-30:]
#
#             lang_code = language_map[st.session_state.translate_to]
#
#             if len(sequence) == 30 and (time.time() - st.session_state.last_prediction_time) > prediction_interval:
#                 # Make a prediction with the model
#                 res = model.predict(np.expand_dims(sequence, axis=0))[0]
#
#                 predictions.append(res)
#
#                 # avg_res = np.mean(predictions, axis=0)
#                 # max_idx = np.argmax(avg_res)
#                 max_idx = np.argmax(res)
#                 # if avg_res[max_idx] > threshold:
#
#                 # if not sentence or (sentence and predicted_class_label != sentence[-1]):
#                 #     sentence.append(predicted_class_label)
#                 #
#                 #     # Translation and text-to-speech (if enabled)
#                 #     original_text = "\n".join(sentence[:-1]) + (
#                 #         " " + sentence[-1] if sentence[:-1] else sentence[-1])
#                 #
#                 #     if lang_code != "en":
#                 #         translated_text = translate_text(original_text, lang_code)
#                 #     else:
#                 #         translated_text = original_text
#                 #
#                 #     if st.session_state.voice_output:
#                 #         if speak_thread is None or not speak_thread.is_alive():
#                 #             speak_thread = threading.Thread(target=speak_word, args=(translated_text, lang_code))
#                 #             speak_thread.start()
#                 #
#                 #     # Update the history
#                 #     update_history(st.session_state.history, sentence, lang_code)
#                 #     st.session_state.last_prediction_time = time.time()
#
#
#                 # Update the history
#                 # speak_thread = threading.Thread(target=speak_word, args=(translated_text, lang_code))
#                 # speak_thread.start()
#                 word = actions[max_idx]
#                 translated_word = translate_text(word, lang_code)
#                 generate_audio_file(word, lang_code, translated_word)
#
#                 thread = threading.Thread(target=speak_word, args=(word, lang_code))
#                 thread.start()
#                 print(word)
#                 word = word.replace("_", " ")
#                 update_history(st.session_state.history, translated_word, lang_code)
#                 st.session_state.last_prediction_time = time.time()
#
#
#             # Display the confidence level
#             # if max_idx is not None:
#             #     confidence_placeholder.markdown(
#             #         f"<h3 style='color: red; font-family: AllenSans, sans-serif;'>Confidence: {avg_res[max_idx]:.2f}</h3>",
#             #         unsafe_allow_html=True)
#             if max_idx is not None:
#                 confidence_placeholder.markdown(
#                     f"<h3 style='color: red; font-family: AllenSans, sans-serif;'>Detected action: {actions[max_idx]}</h3>",
#                     unsafe_allow_html=True)
#
#             # Display the frame in the Streamlit app
#             camera_window.image(image, channels='BGR')
#
#             # Display the history of recognized signs
#             history_text = "\n".join(st.session_state.history)
#             history_placeholder.markdown(f"<div class='history-text'>{history_text}</div>", unsafe_allow_html=True)
#     predictions.clear()
#
#     cap.release()
#     cv2.destroyAllWindows()
