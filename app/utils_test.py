import tensorflow as tf
from typing import List
import mediapipe as mp
import cv2
import os 
import re

vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
num_to_char = tf.keras.layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True)

def load_video(path:str) -> List[float]:
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh()
    #print(f"Processing video: {path}")  # Print out the path of the video being processed
    cap = cv2.VideoCapture(path)
    frames=[]
    lips=[216, 430]

    try:
        for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
            ret, frame = cap.read()
            height, width, _ = frame.shape

            # Check if the input video is RGB, if not, convert it to RGB
            if frame.shape[2] != 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            result = face_mesh.process(frame)
            if result is not None and result.multi_face_landmarks:
                try:
                    for facial_landmarks in result.multi_face_landmarks:
                        x_max = 0
                        y_max = 0
                        x_min = width
                        y_min = height
                        for fl in lips:
                            pt1 = facial_landmarks.landmark[fl]
                            x, y = int(pt1.x * width), int(pt1.y * height)
                            if x > x_max:
                                x_max = x
                            if x < x_min:
                                x_min = x
                            if y > y_max:
                                y_max = y
                            if y < y_min:
                                y_min = y
                        gr = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                        cropped_frame = gr[y_min:y_max, x_min:x_max]
                        cropped_frame = cv2.resize(cropped_frame, (140, 46))
                        frames.append(cropped_frame)
                except Exception as t:
                    print(f"Error load_video at path {path}: {t}")
                    return None

    except Exception as e:
        print(f'Error processing video: {path}: {e}')  # Print out the error if any occurs during video processing
        return None
    finally:
        cap.release()
    mean = tf.math.reduce_mean(frames)
    std = tf.math.reduce_std(tf.cast(frames, tf.float32))
    return tf.cast((frames - mean), tf.float32) / std


def load_data_test(path: str):
    path = bytes.decode(path.numpy())
    #file_name = path.split('/')[-1].split('.')[0] # Linux
    file_name = path.split('\\')[-1].split('.')[0] # Windows
    match = re.search(r's(\d+)', path)
    number = match.group(1)

    global video_path
    if path.lower().endswith('.mpg'):
        video_path = os.path.join('..','data',f's{number}',f'{file_name}.mpg')
    else:
        video_path = os.path.join('..','data',f's{number}',f'{file_name}.mp4')
    try:
        frames = load_video(video_path)
        #print(f"Frames shape: {frames.shape}")
        return frames

    except Exception as e:
        #print(f'Error load_data at path {path}: {e}')
        return None