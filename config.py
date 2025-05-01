import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

UPLOAD_FOLDER = Path(os.getenv('UPLOAD_FOLDER'))
DEBUG = os.getenv('DEBUG') == 'True'
