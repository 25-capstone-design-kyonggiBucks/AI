import os
import cv2
import json
import numpy as np
from tqdm import tqdm
from  multi import overlay_face_on_placeholder  # 실제 경로로 변경
video_path = '/Users/byeonjuhyeong/Desktop/uploads/videos/default/axe.mp4'
output_json = '/Users/byeonjuhyeong/Desktop/uploads/bbox_annotation.json'

def generate_bbox_annotations(video_path, output_json):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    annotations = {}
    last_bbox = None

    for frame_idx in tqdm(range(total_frames), desc="Generating BBoxes"):
        ret, frame = cap.read()
        if not ret:
            break

        # bbox만 얻기 위한 호출
        _, bbox = overlay_face_on_placeholder(frame, np.zeros((100, 100, 3), dtype=np.uint8), frame_idx, last_bbox)
        if bbox is not None:
            x, y, w, h = bbox
            annotations[str(frame_idx)] = {"bbox": [int(x), int(y), int(w), int(h)], "valid": True}
            last_bbox = bbox
        else:
            annotations[str(frame_idx)] = {"bbox": None, "valid": False}
            last_bbox = None

    with open(output_json, "w") as f:
        json.dump(annotations, f, indent=2)

    cap.release()
    print(f"✅ bbox annotation saved to: {output_json}")

# 실행
generate_bbox_annotations(video_path, output_json)
