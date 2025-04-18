import cv2
import numpy as np
from deepface import DeepFace
import os

def extract_and_save_face(img_path, emotion, output_dir="/Users/kim-woohyeon/Desktop"):
    """
    이미지에서 얼굴을 추출하고 주어진 감정(emotion)에 따라 파일명으로 저장하는 함수.

    Parameters:
        img_path (str): 입력 이미지 경로
        emotion (str): 저장할 감정 라벨 (예: happy, sad, surprised 등)
        output_dir (str): 출력 디렉토리 경로 (기본값은 데스크탑)
    
    Returns:
        saved_path (str or None): 저장된 파일 경로 또는 None (얼굴이 없을 경우)
    """
    try:
        # 얼굴 검출
        faces = DeepFace.extract_faces(img_path, detector_backend="retinaface", enforce_detection=True)

        if not faces:
            print("얼굴이 검출되지 않았습니다.")
            return None

        # 첫 번째 얼굴만 사용
        face_data = faces[0]
        face = face_data["face"]

        # 이미지 저장 형식으로 변환
        face_uint8 = (face * 255).astype(np.uint8)
        face_bgr = cv2.cvtColor(face_uint8, cv2.COLOR_RGB2BGR)

        # 파일 이름 지정 (emotion 값으로)
        file_name = f"{emotion}.jpg"
        output_path = os.path.join(output_dir, file_name)

        # 저장
        cv2.imwrite(output_path, face_bgr)
        print(f"[{emotion}] 정렬된 얼굴 이미지가 {output_path}에 저장되었습니다.")

        return output_path

    except Exception as e:
        print(f"오류 발생: {e}")
        return None

if __name__ == "__main__":
    # 테스트용 경로와 감정 입력
    test_img_path = "/Users/kim-woohyeon/Desktop/pph.jpg"
    test_emotion = "happy"

    # 함수 실행
    extract_and_save_face(test_img_path, test_emotion)