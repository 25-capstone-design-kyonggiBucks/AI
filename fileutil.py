import os
from pathlib import Path
from config import UPLOAD_FOLDER, DEBUG

# config에 정의돈 UPLDOADFOLDER + rquest_path
def resolve_uploaded_file_path(request_path: str):

    request_path = request_path.lstrip('/')

    full_path = (UPLOAD_FOLDER / request_path).resolve()

        # 보안 체크: 업로드 폴더 내부에 있는지
    if not str(full_path).startswith(str(UPLOAD_FOLDER)):
        raise PermissionError(f"[ERROR] 보안 위반 경로\n: {full_path}")

    if not full_path.exists() or not full_path.is_file():
        raise FileNotFoundError(f"[ERROR] 파일 없음\n: {full_path}")

    return full_path

        

def main():
    # 테스트 경로
    test_paths = [
        "/uploads/images/Golden_axe,_silver_axe.jpg", # 성공 테스트
        "/uploads/videos/default/axe.mp4", # 성공 테스트
        "/uploads/images/The_giving_tree.jpg", # 성공 테스트
        "/etc/passwd",  # 예외 테스트
    ]

    for path in test_paths:
        try:
            result = resolve_uploaded_file_path(path)
            print(f"[success] {path} -----> {result}")
        except Exception as e:
            print(f"[fail] {path} ----> 오류: {e}")


if __name__ == "__main__":
    main()