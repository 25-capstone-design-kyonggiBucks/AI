import os        # 파일 경로 조작 (os.path.join 등)
import cv2       # OpenCV (cv2.imread 등)

face_cache = {}

def get_face_image(expr, fallback, expressions_dir,shared_cache=None):
    if expr in face_cache:
        return face_cache[expr]
    img_path = os.path.join(expressions_dir, f"{expr}.jpg")
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        img = fallback
    if shared_cache is not None:
        shared_cache[expr] = img
    face_cache[expr] = img
    return img
