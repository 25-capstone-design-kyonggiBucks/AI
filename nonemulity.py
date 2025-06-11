import os
import cv2
import json
import numpy as np
import cashUtil
import time
from multiprocessing import Process, Queue, cpu_count,SimpleQueue
from threading import Thread

def overlay_face_on_placeholde2(frame, face_img, frame_idx=None,bbox=None):
    # 8) bbox 유효성 검사
    if bbox is None:
        return frame
    else:
        last_bbox = bbox
        x0, y0, w, h = map(int,bbox)

    # 9) 얼굴 오버레이
    face_w = face_h = int(w * 0.9)
    face_resized = cv2.resize(face_img, (face_w, face_h), interpolation=cv2.INTER_AREA)
    offset_x = x0 + (w - face_w) // 2
    offset_y = y0 + (h - face_h) // 2

    mask = np.zeros((face_h, face_w), dtype=np.uint8)
    cv2.circle(mask, (face_w // 2, face_h // 2), face_w // 2, 255, -1)
    mask = cv2.GaussianBlur(mask, (21,21), 10)
    mask_f = mask.astype(np.float32) / 255.0
    mask_3c = cv2.merge([mask_f]*3)

    if face_resized.shape[2] == 4:
        face_rgb = face_resized[:, :, :3]
    else:
        face_rgb = face_resized

    roi = frame[offset_y:offset_y + face_h, offset_x:offset_x + face_w]
    if roi.shape[:2] != face_rgb.shape[:2]:
        return frame

    blended = (roi*(1-mask_3c) + face_rgb*mask_3c).astype(np.uint8)
    frame[offset_y:offset_y + face_h, offset_x:offset_x + face_w] = blended
    return frame


def process_video(video_path, json_path, expressions_dir, fallback_img, output_path,bbox_path):
    """
    비디오 처리 함수
    
    Args:
        video_path (str): 입력 비디오 경로
        json_path (str): 애노테이션 JSON 경로
        expressions_dir (str): 표정 이미지가 있는 디렉토리 경로
        fallback_img (str): 기본 이미지 경로
        output_path (str): 출력 비디오 경로
        
    Returns:
        str: 출력 비디오 경로
    """
    
    # 경로 정규화
    video_path = os.path.normpath(video_path)
    json_path = os.path.normpath(json_path)
    expressions_dir = os.path.normpath(expressions_dir)
    fallback_img = os.path.normpath(fallback_img)
    output_path = os.path.normpath(output_path)
    
    # 디버깅을 위한 경로 출력
    print(f"입력 비디오 경로: {video_path}")
    print(f"출력 비디오 경로: {output_path}")
    
    # 출력 디렉토리 생성
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # 애노테이션 JSON 로드
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            annotations = json.load(f)
        print(f"Loaded {len(annotations)} annotations.")
    else:
        annotations = {}
        print("No annotation.json → all frames face=False")

    # fallback 이미지 로드
    fallback = cv2.imread(fallback_img, cv2.IMREAD_UNCHANGED)
    if fallback is None:
        raise FileNotFoundError(f"Fallback image not found: {fallback_img}")

    # 비디오 열기
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    # 출력 비디오 세팅
    fps    = cap.get(cv2.CAP_PROP_FPS)
    W      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 웹 브라우저와 호환되는 코덱 사용
    # mp4v 대신 H.264 코덱 사용
    try:
        # H.264 코덱 시도
        fourcc = cv2.VideoWriter_fourcc(*'H264')
        print("H264 코덱 사용")
    except:
        # 실패하면 기본 mp4v 코덱 사용
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        print("mp4v 코덱 사용 (웹 브라우저 호환성 제한)")
    
    out    = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

    # bbox.json 로딩
    if os.path.exists(bbox_path):
        with open(bbox_path, 'r') as f:
            bbox_data = json.load(f)
        print(f"Loaded {len(bbox_data)} bbox_data")
    else:
        raise FileNotFoundError(f" bbox 파일을 찾을 수 없습니다: {bbox_path}")

    start_time = time.time()

    # 프레임 루프
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        ann = annotations.get(str(frame_idx), {"face": False, "expression": None})
        bbox_entry = bbox_data.get(str(frame_idx), {"bbox": None, "valid": False})

        if not ann["face"] or not bbox_entry["valid"]:
            out_frame = frame
        else:
            expr = ann["expression"] or "base"
            face_img = cashUtil.get_face_image(expr,fallback,expressions_dir)
            out_frame = overlay_face_on_placeholde2(frame, face_img, frame_idx,bbox_entry["bbox"])

        out.write(out_frame)
        frame_idx += 1
        print(frame_idx)

    duration = time.time() - start_time

    cap.release()
    out.release()
    print("✅ Done →", output_path)
    print("시간",duration)
    
    # 경로 일관성을 위해 슬래시를 사용하는 URL 형식으로 반환
    return output_path.replace("\\", "/")

# 테스트용 메인 함수
if __name__ == "__main__":
    # 설정 경로 및 파일
    video_path      = '/Users/byeonjuhyeong/Desktop/uploads/videos/default/axe.mp4'
    json_path       = '/Users/byeonjuhyeong/Desktop/uploads/annotation.json'
    expressions_dir = '/Users/byeonjuhyeong/Desktop/uploads/images'         # base.jpg, sad.jpg, mad.jpg, smile.jpg
    fallback_img    = '/Users/byeonjuhyeong/Desktop/uploads/images/base.jpg'
    output_path     = '/Users/byeonjuhyeong/Desktop/uploads/videos/custom/test_output.mp4'
    bbox_path = "/Users/byeonjuhyeong/Desktop/uploads/bbox_annotation.json"
    
    process_video(video_path, json_path, expressions_dir, fallback_img, output_path,bbox_path)


