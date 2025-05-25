import shutil
import os
import config
import forVideo
import uuid

import os
import shutil
import uuid  # ✅ 누락되었던 uuid import

def copy_video(video_path, output_path):
    # 파일 존재 여부 확인
    if not os.path.isfile(video_path):
        print("지정한 경로에 파일이 존재하지 않습니다.")
        return
    print("지정한 경로에 파일 존재")

    # 파일 이름 추출 및 새 이름 생성
    original_filename = os.path.basename(video_path)
    print(f"원본 파일명: {original_filename}")
    filename = create_fileName(original_filename)
    
    # 저장할 경로 생성
    base, ext = os.path.splitext(filename)
    copy_filename = f"{base}_copy{ext}"

    # 출력 폴더가 없으면 생성
    os.makedirs(output_path, exist_ok=True)

    # 최종 복사 경로 (output_path 기준)
    copy_path = os.path.join(output_path, copy_filename)

    # 복사 수행
    shutil.copy(video_path, copy_path)
    print(f"복사 완료: {copy_path}")

    video_url = os.path.join('/uploads/videos/custom', copy_filename)
    return video_url

def create_fileName(original_filename):
    name_part, ext_part = os.path.splitext(original_filename)
    unique_id = str(uuid.uuid4())
    new_filename = f"{unique_id}_{original_filename}"
    print(f"생성된 파일명: {new_filename}")
    return new_filename

def main():
    video_path = '/Users/byeonjuhyeong/Desktop/uploads/videos/default/toystory.mp4'
    output_path = '/Users/byeonjuhyeong/Desktop/uploads/videos/custom'
    video_url = copy_video(video_path, output_path)

    print("video_url = "+ video_url)

if __name__ == "__main__":
    main()
