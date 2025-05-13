import cv2
import numpy as np
from deepface import DeepFace
import os
import uuid
import platform

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


def extract_and_save_face2(img, output_path):
    try:
        # output_path 디버깅 정보 추가
        print(f"저장 경로: {output_path}")
        
        # 경로 존재 확인 및 디렉토리 생성
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f"디렉토리 생성: {output_dir}")
        
        # raw 이미지 데이터
        face_bgr = extract_faceImage(img)
        
        # 이미지 저장
        save_faceImage(output_path, face_bgr)
        
        # 저장 후 파일 존재 확인
        if os.path.exists(output_path):
            print(f"파일 저장 확인: {output_path} (크기: {os.path.getsize(output_path)} bytes)")
        else:
            print(f"경고: 파일이 저장되지 않았습니다! 경로: {output_path}")

    except Exception as e:
        print(f"오류 발생 (extract_and_save_face2): {str(e)}")
        raise


def extract_faceImage(img):
    try:
        # 얼굴 검출 (Numpy array를 직접 사용)
        faces = DeepFace.extract_faces(img_path=img, detector_backend="retinaface", enforce_detection=True)

        if not faces:
            raise ValueError("[ERROR] 얼굴이 검출되지 않았습니다.")

        face_data = faces[0]
        face = face_data["face"]

        face_uint8 = (face * 255).astype(np.uint8)
        face_bgr = cv2.cvtColor(face_uint8, cv2.COLOR_RGB2BGR)

        return face_bgr
    except Exception as e:
        print(f"오류 발생 (extract_faceImage): {str(e)}")
        raise

def save_faceImage(output_path, face_bgr):
    try:
        # 경로 정규화 (OS별 경로 구분자 처리)
        output_path = os.path.normpath(output_path)
        
        # 디렉토리 확인
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            
        # 이미지 저장
        success = cv2.imwrite(output_path, face_bgr)
        
        if success:
            print(f"정렬된 얼굴 이미지가 {output_path}에 저장되었습니다.")
        else:
            print(f"이미지 저장 실패: {output_path}")
            raise IOError(f"이미지 저장 실패: {output_path}")
            
    except Exception as e:
        print(f"오류 발생 (save_faceImage): {str(e)}")
        raise

def convert_file_to_cv2_image(image_file):
    try:
        file_bytes = np.frombuffer(image_file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("이미지를 디코딩할 수 없습니다")
        return img
    except Exception as e:
        print(f"이미지 변환 오류: {str(e)}")
        raise
    
def create_fileName(original_filename, emotion):
    # 확장자 분리
    name_part, ext_part = os.path.splitext(original_filename)
    
    # UUID 생성 - 스프링 서버와 동일한 형식으로
    unique_id = str(uuid.uuid4())
    
    # 스프링 서버와 호환되는 파일명 형식
    new_filename = f"{unique_id}_{original_filename}"
    
    # 디버깅 정보
    print(f"생성된 파일명: {new_filename}")
    
    return new_filename


if __name__ == "__main__":
    # 테스트용 경로와 감정 입력
    test_img_path = "/Users/kim-woohyeon/Desktop/pph.jpg"
    test_emotion = "happy"

    # 함수 실행
    extract_and_save_face(test_img_path, test_emotion)