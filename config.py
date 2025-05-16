import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# 사용자 홈 디렉토리 경로 가져오기
USER_HOME = os.path.expanduser("~")

# 기본값은 스프링 서버와 같은 경로로 설정
DEFAULT_UPLOAD_FOLDER = os.path.join(USER_HOME, "Desktop")

# .env 파일에서 설정을 가져옴
UPLOAD_FOLDER = Path(os.getenv('UPLOAD_FOLDER', DEFAULT_UPLOAD_FOLDER))

DEBUG = os.getenv('DEBUG', 'True') == 'True'

# 디버깅 정보 출력
print(f"사용 중인 업로드 폴더: {UPLOAD_FOLDER}")