import os
import cv2
import json
import numpy as np
from tqdm import tqdm
import  customForVideo
import cashUtil


video_path = '/Users/byeonjuhyeong/Desktop/uploads/videos/default/axe.mp4'
output_json = '/Users/byeonjuhyeong/Desktop/uploads/bbox_annotation.json'
expressions_dir = '/Users/byeonjuhyeong/Desktop/uploads/images'         # base.jpg, sad.jpg, mad.jpg, smile.jpg
fallback_img    = '/Users/byeonjuhyeong/Desktop/uploads/images/base.jpg'
json_path       = '/Users/byeonjuhyeong/Desktop/uploads/annotation.json'


DEBUG = False  # 디버깅 이미지 저장 여부
def generate_bbox_annotations(video_path, output_json, fallback, expressions_dir, annotation_json):


    # 애노테이션 로딩
    if not os.path.exists(annotation_json):
        raise FileNotFoundError(f"Annotation JSON not found: {annotation_json}")
    with open(annotation_json, 'r') as f:
        annotations_data = json.load(f)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    bbox_annotations = {}
    last_bbox = None

    for frame_idx in tqdm(range(total_frames), desc="Generating BBoxes"):
        ret, frame = cap.read()
        if not ret:
            break

        ann = annotations_data.get(str(frame_idx), {"face": False, "expression": None})

        if not ann["face"]:
            bbox_annotations[str(frame_idx)] = {"bbox": None, "valid": False}
            last_bbox = None
            continue

        expr = ann["expression"] or "base"
        face_img = cashUtil.get_face_image(expr, fallback, expressions_dir)

        # bbox만 얻기 위해 frame 전달
        _, last_bbox = overlay_face_on_placeholder(frame, face_img, frame_idx,last_bbox)
        if last_bbox is not None:
            x, y, w, h = last_bbox
            bbox_annotations[str(frame_idx)] = {"bbox": [int(x), int(y), int(w), int(h)], "valid": True}
        else:
            bbox_annotations[str(frame_idx)] = {"bbox": None, "valid": False}

    cap.release()

    # 저장
    with open(output_json, "w") as f:
        json.dump(bbox_annotations, f, indent=2)

    print(f"✅ bbox annotation saved to: {output_json}")


def overlay_face_on_placeholder(frame, face_img, frame_idx=None,last_bbox=None):
    """
    1) HSV 임계값 완화 + 마스크 팽창(dilate)
    2) HoughLinesP 파라미터 완화
    3) vlen/hlen 중 더 긴 쪽을 사용해 '의사 반지름(r)' 계산
    4) 새 중심(cx, cy)과 이전 중심(px, py) 간 이동 거리(d)를 계산
       → d > MAX_SHIFT(r-based) 이면 이전 박스 재사용
    5) r*0.9 배율로 얼굴 크기(정사각형) 설정
    """

    Hf, Wf = frame.shape[:2]

    # 1) HSV 변환
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 2) 빨간색 마스크 (Hue 0~20, Sat>=50, Val>=50)
    lower_red1 = np.array([0,  50,  50])
    upper_red1 = np.array([20, 255, 255])
    lower_red2 = np.array([160, 50, 50])
    upper_red2 = np.array([180,255,255])
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    # 3) 파란색 마스크 (Hue 90~140, Sat>=50, Val>=50)
    lower_blue = np.array([90,  50,  50])
    upper_blue = np.array([140, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    # 4) 마스크 팽창(dilate) → 선 굵기 보정
    kernel = np.ones((3, 3), np.uint8)
    mask_red   = cv2.dilate(mask_red,  kernel, iterations=1)
    mask_blue  = cv2.dilate(mask_blue, kernel, iterations=1)

    if DEBUG and frame_idx is not None:
        os.makedirs("debug", exist_ok=True)
        cv2.imwrite(f"debug/frame_{frame_idx:04d}_mask_red.png",  mask_red)
        cv2.imwrite(f"debug/frame_{frame_idx:04d}_mask_blue.png", mask_blue)

    # 5) Canny → HoughLinesP: "빨간(수직)선" 검출
    edges_red = cv2.Canny(mask_red, 50, 150)
    lines_red = cv2.HoughLinesP(
        edges_red,
        rho=1,
        theta=np.pi/180,
        threshold=30,
        minLineLength=Hf // 16,
        maxLineGap=10
    )
    vertical_line = None
    if lines_red is not None:
        max_len = 0
        for x1, y1, x2, y2 in lines_red[:, 0]:
            dx = x2 - x1
            dy = y2 - y1
            length = np.hypot(dx, dy)
            if abs(dx) < abs(dy) and length > max_len:
                max_len = length
                vertical_line = (x1, y1, x2, y2, length)

    # 6) Canny → HoughLinesP: "파란(수평)선" 검출
    edges_blue = cv2.Canny(mask_blue, 50, 150)
    lines_blue = cv2.HoughLinesP(
        edges_blue,
        rho=1,
        theta=np.pi/180,
        threshold=30,
        minLineLength=Wf // 16,
        maxLineGap=10
    )
    horizontal_line = None
    if lines_blue is not None:
        max_len = 0
        for x1, y1, x2, y2 in lines_blue[:, 0]:
            dx = x2 - x1
            dy = y2 - y1
            length = np.hypot(dx, dy)
            if abs(dy) < abs(dx) and length > max_len:
                max_len = length
                horizontal_line = (x1, y1, x2, y2, length)

    if DEBUG and frame_idx is not None:
        debug_frame = frame.copy()
        if vertical_line is not None:
            x1, y1, x2, y2, _ = vertical_line
            cv2.line(debug_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        if horizontal_line is not None:
            x1, y1, x2, y2, _ = horizontal_line
            cv2.line(debug_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.imwrite(f"debug/frame_{frame_idx:04d}_lines.png", debug_frame)

    # 7) 교차점 계산 및 의사 반지름(r) 계산
    if vertical_line is not None and horizontal_line is not None:
        vx1, vy1, vx2, vy2, vlen = vertical_line
        hx1, hy1, hx2, hy2, hlen = horizontal_line

        cx = int((vx1 + vx2) / 2)
        cy = int((hy1 + hy2) / 2)

        # 최소 선 길이 기준
        MIN_LINELEN = Hf // 10
        if vlen < MIN_LINELEN or hlen < MIN_LINELEN:
            bbox = last_bbox
        else:
            diameter_est = max(vlen, hlen) * 1.2
            r = int(diameter_est / 2)

            if last_bbox is not None:
                px = last_bbox[0] + last_bbox[2] // 2
                py = last_bbox[1] + last_bbox[3] // 2
                d = np.hypot(cx - px, cy - py)
                MAX_SHIFT = r * 1.5
                if d > MAX_SHIFT:
                    bbox = last_bbox
                else:
                    x0 = cx - r
                    y0 = cy - r
                    w  = h  = 2 * r
                    x0 = max(0, min(Wf - w, x0))
                    y0 = max(0, min(Hf - h, y0))
                    bbox = (x0, y0, w, h)
            else:
                x0 = cx - r
                y0 = cy - r
                w  = h  = 2 * r
                x0 = max(0, min(Wf - w, x0))
                y0 = max(0, min(Hf - h, y0))
                bbox = (x0, y0, w, h)
    else:
        bbox = last_bbox

    if DEBUG and frame_idx is not None:
        debug_frame2 = frame.copy()
        if bbox is not None:
            x0, y0, w, h = bbox
            cv2.rectangle(debug_frame2, (x0, y0), (x0 + w, y0 + h), (0, 255, 0), 2)
            cv2.imwrite(f"debug/frame_{frame_idx:04d}_bbox.png", debug_frame2)

    # 8) bbox 유효성 검사
    if bbox is None:
        last_bbox = None
        return frame,last_bbox
    else:
        last_bbox = bbox
        x0, y0, w, h = bbox

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
        return frame,last_bbox

    blended = (roi*(1-mask_3c) + face_rgb*mask_3c).astype(np.uint8)
    frame[offset_y:offset_y + face_h, offset_x:offset_x + face_w] = blended
    return frame,last_bbox

# 실행
generate_bbox_annotations(video_path, output_json,fallback_img,expressions_dir,json_path)
