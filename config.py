import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# 사용자 홈 디렉토리 경로 가져오기
USER_HOME = os.path.expanduser("~")

# 기본값은 스프링 서버와 같은 경로로 설정
DEFAULT_UPLOAD_FOLDER = os.path.join(USER_HOME, "Desktop", "uploads")

# .env 파일에서 설정을 가져옴
env_upload_folder = os.getenv('UPLOAD_FOLDER')
print(f"환경 변수 UPLOAD_FOLDER: {env_upload_folder}")

# 환경 변수가 설정되어 있으면 그 값을 사용, 아니면 기본값 사용
if env_upload_folder:
    UPLOAD_FOLDER = Path(env_upload_folder)
else:
    UPLOAD_FOLDER = Path(DEFAULT_UPLOAD_FOLDER)

DEBUG = os.getenv('DEBUG', 'True') == 'True'

# 디버깅 정보 출력
print(f"사용 중인 업로드 폴더: {UPLOAD_FOLDER}")