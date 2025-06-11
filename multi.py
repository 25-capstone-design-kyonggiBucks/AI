import os
import cv2
import json
import numpy as np
import cashUtil
import time
from multiprocessing import Process, Queue, cpu_count,SimpleQueue
from threading import Thread

# ——————————————————————————————
# 설정 경로 및 파일 (테스트용)
video_path      = '/Users/byeonjuhyeong/Desktop/uploads/videos/default/axe.mp4'
json_path       = '/Users/byeonjuhyeong/Desktop/uploads/annotation.json'
expressions_dir = '/Users/byeonjuhyeong/Desktop/uploads/images'         # base.jpg, sad.jpg, mad.jpg, smile.jpg
fallback_img    = 'C:/Users/winte/Desktop/uploads/images/happy.jpg'
output_path     = 'C:/Users/winte/Desktop/uploads/videos/custom/test_output.mp4'
# ——————————————————————————————


# 디버그 모드: True면 debug/ 폴더에 이미지 저장
DEBUG = False

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
        return frame
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
        return frame

    blended = (roi*(1-mask_3c) + face_rgb*mask_3c).astype(np.uint8)
    frame[offset_y:offset_y + face_h, offset_x:offset_x + face_w] = blended
    return frame, last_bbox

def overlay_face_on_placeholde2(frame, face_img, frame_idx=None,bbox=None):
    # 8) bbox 유효성 검사
    if bbox is None:
        return frame
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

    
    q_left  = Queue(maxsize=32)
    q_mid = Queue(maxsize=32)
    q_right = Queue(maxsize=32)

    q2 = Queue(maxsize=1000)

    reader1 = Thread(target=read_frames, args=(cap, annotations, q_left,q_mid,q_right))


    processor = Process(target=process_frames, args=(q_left, q2, expressions_dir, fallback,bbox_data))
    processor2 = Process(target=process_frames, args=(q_mid, q2, expressions_dir, fallback,bbox_data))
    processor3 = Process(target=process_frames, args=(q_right, q2, expressions_dir, fallback,bbox_data))
    writer = Thread(target=write_frames, args=(q2, out, 3))

    start_time = time.time()
    reader1.start()
    processor.start()
    processor2.start()
    processor3.start()
    writer.start()

    reader1.join()
    processor.join()
    processor2.join()
    processor3.join()
    writer.join()
    duration = time.time() - start_time

    cap.release()
    out.release()
    print("✅ Done →", output_path)
    print("시간",duration)
    
    # 경로 일관성을 위해 슬래시를 사용하는 URL 형식으로 반환
    return output_path.replace("\\", "/")

def read_frames(cap, annotations, q_left, q_mid, q_right):
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        ann = annotations.get(str(frame_idx), {"face": False, "expression": None})

        if frame_idx % 3 == 0:
            q_left.put((frame_idx, frame, ann))
        elif frame_idx % 3 == 1:
            q_mid.put((frame_idx, frame, ann))
        else:
            q_right.put((frame_idx, frame, ann))

        frame_idx += 1

    # 종료 신호 전송
    q_left.put(None)
    q_mid.put(None)
    q_right.put(None)

            
def process_frames(queue_in, queue_out, expressions_dir, fallback,bbox_data):
    while True :
        item = queue_in.get()
        if item is None:
            queue_out.put(None)
            break
        frame_idx, frame, ann = item
        bbox_entry = bbox_data.get(str(frame_idx), {"bbox": None, "valid": False})

        if not ann["face"] or not bbox_entry["valid"]:
            out_frame = frame
        else:
            expr = ann["expression"] or "base"
            face_img = cashUtil.get_face_image(expr,fallback,expressions_dir)
            out_frame = overlay_face_on_placeholde2(frame, face_img, frame_idx,bbox_entry["bbox"])
        queue_out.put((frame_idx, out_frame))
        print(frame_idx)

def write_frames(queue, out,num_workers):
    buffer = {}
    next_idx = 0
    end_signals = 0
    while True:
        item = queue.get()
        if item is None:
            end_signals += 1
            if end_signals == num_workers: ## 처리 프로세스 개수
                break
            continue
        frame_idx,out_frame = item
        buffer[frame_idx] = out_frame
        while next_idx in buffer:
            out.write(buffer.pop(next_idx))
            next_idx += 1
    
    


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
