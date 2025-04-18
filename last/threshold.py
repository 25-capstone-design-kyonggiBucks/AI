import pandas as pd
import numpy as np

def calculate_threshold(csv_file, center_x, range_width=5):
    """
    특정 중심값에서 일정 범위의 임계값을 계산하는 함수.
    
    Parameters:
        csv_file (str): CSV 파일 경로.
        center_x (float): 임계값을 계산할 중심 X 값.
        range_width (float): X 값 범위 폭 (기본값은 중심값을 기준으로 ±5).

    Returns:
        dict: 범위 내 평균과 표준편차를 기반으로 한 임계값.
    """
    # CSV 파일 읽기
    data = pd.read_csv(csv_file, names=['X', 'Y'])
    
    # 지정된 범위 내에 있는 데이터 필터링
    min_x = center_x - range_width
    max_x = center_x + range_width
    filtered_data = data[(data['X'] >= min_x) & (data['X'] <= max_x)]
    
    # 임계값 계산 (평균 ± 표준편차)
    mean_value = filtered_data['Y'].mean()
    std_dev = filtered_data['Y'].std()
    
    threshold = {
        'mean': mean_value,
        'lower_threshold': mean_value - std_dev,
        'upper_threshold': mean_value + std_dev
    }
    
    return threshold

# 사용 예시
if __name__ == "__main__":
    csv_file_path = r'E:\dm-woohyeon\concrete_w\data\air1.csv'  # CSV 파일 경로

    # 80 근처의 임계값 계산
    threshold_80 = calculate_threshold(csv_file_path, center_x=80)
    print(f"80 근처에서의 임계값:")
    print(f"평균값 (Mean): {threshold_80['mean']}")
    print(f"하한 임계값 (Lower Threshold): {threshold_80['lower_threshold']}")
    print(f"상한 임계값 (Upper Threshold): {threshold_80['upper_threshold']}\n")

    # 100 근처의 임계값 계산
    threshold_100 = calculate_threshold(csv_file_path, center_x=100)
    print(f"100 근처에서의 임계값:")
    print(f"평균값 (Mean): {threshold_100['mean']}")
    print(f"하한 임계값 (Lower Threshold): {threshold_100['lower_threshold']}")
    print(f"상한 임계값 (Upper Threshold): {threshold_100['upper_threshold']}")
