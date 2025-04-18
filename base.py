import cv2
import numpy as np
from deepface import DeepFace

# 입력 이미지 경로 설정
img_path = "/Users/kim-woohyeon/Desktop/pph.jpg"

# DeepFace를 사용하여 얼굴 추출 (추출 결과는 얼굴 정보를 담은 dict 리스트)
faces = DeepFace.extract_faces(img_path, detector_backend="retinaface", enforce_detection=True)

# 얼굴이 하나라도 검출되었는지 확인
if len(faces) == 0:
    print("얼굴이 검출되지 않았습니다.")
else:
    # 첫 번째 얼굴 추출 (dict 형태로 반환됨)
    # 각 dict에는 "face", "facial_area", "confidence" 등이 포함됩니다.
    face_data = faces[0]
    face = face_data["face"]

    # DeepFace는 얼굴 이미지를 0~1 범위의 float 배열로 반환하므로, 
    # 저장을 위해 0~255 범위의 uint8 타입으로 변환하고 RGB->BGR 변환
    face_uint8 = (face * 255).astype(np.uint8)
    face_bgr = cv2.cvtColor(face_uint8, cv2.COLOR_RGB2BGR)

    # 결과 이미지 저장 (원하는 경로로 수정)
    output_path = "/Users/kim-woohyeon/Desktop/aligned_face.jpg"
    cv2.imwrite(output_path, face_bgr)
    print(f"정렬된 얼굴 이미지가 {output_path}에 저장되었습니다.")
