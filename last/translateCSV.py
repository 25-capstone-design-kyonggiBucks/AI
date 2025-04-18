import os
import pandas as pd
import argparse

# ArgumentParser 객체 생성
parser = argparse.ArgumentParser(description="Swap columns X and Y in all CSV files within a directory and save to another directory.")
parser.add_argument('input_dir', type=str, help='Path to the input directory containing CSV files')
parser.add_argument('output_dir', type=str, help='Path to the output directory where modified CSV files will be saved')
args = parser.parse_args()

# 출력 디렉토리 생성 (존재하지 않는 경우)
os.makedirs(args.output_dir, exist_ok=True)

# 디렉토리 내의 모든 CSV 파일 처리
for filename in os.listdir(args.input_dir):
    if filename.endswith(".csv"):
        file_path = os.path.join(args.input_dir, filename)
        
        # CSV 파일 로드 (헤더 있음, 첫 번째 행 제거)
        data = pd.read_csv(file_path, header=0)
        
        # X와 Y 컬럼 값들을 서로 바꾸기
        data = data.iloc[:, [1, 0]]
        
        # 새로운 CSV 파일로 저장
        new_file_path = os.path.join(args.output_dir, filename)
        data.to_csv(new_file_path, index=False, header=False)

print(f"모든 CSV 파일의 헤더를 제거하고, X와 Y 컬럼을 바꾸어 새 파일로 저장했습니다. 입력 디렉토리: {args.input_dir}, 출력 디렉토리: {args.output_dir}")
