import cv2
import numpy as np
from deepface import DeepFace
import os

def save_detected_face(frame, emotion_label, output_dir="output", face_index=0):
    try:
        result = DeepFace.extract_faces(frame, detector_backend="retinaface", enforce_detection=True)
        if not result:
            return False

        face_data = result[face_index]["face"]
        face_uint8 = (face_data * 255).astype(np.uint8)
        face_bgr = cv2.cvtColor(face_uint8, cv2.COLOR_RGB2BGR)

        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, f"{emotion_label}_{cv2.getTickCount()}.jpg")
        cv2.imwrite(file_path, face_bgr)
        print(f"[{emotion_label}] 얼굴 저장됨: {file_path}")
        return True
    except Exception as e:
        print(f"저장 실패: {e}")
        return False

def analyze_video_emotions(video_path, threshold_dict=None, frame_interval=30):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    if threshold_dict is None:
        threshold_dict = {"happy": 0.7, "sad": 0.7, "surprise": 0.7}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            try:
                analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)[0]
                dominant_emotion = analysis["dominant_emotion"]
                emotion_scores = analysis["emotion"]

                for emo, thresh in threshold_dict.items():
                    if emotion_scores.get(emo, 0) > thresh:
                        save_detected_face(frame, emo)
            except Exception as e:
                print(f"감정 분석 실패: {e}")
        
        frame_count += 1

    cap.release()
    print("영상 분석 완료.")

if __name__ == "__main__":
    video_path = "/Users/kim-woohyeon/Desktop/sample.mp4"
    analyze_video_emotions(video_path)
