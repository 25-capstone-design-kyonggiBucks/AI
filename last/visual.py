import pandas as pd
import matplotlib.pyplot as plt

# CSV 파일 경로
input_file = r"E:\dm-woohyeon\concrete_w\data\air1.csv"

# CSV 파일을 pandas DataFrame으로 읽기 (첫 번째 행을 열 이름으로 사용하지 않음)
df = pd.read_csv(input_file, header=None, names=['X', 'Y'])
smoothed_data = df.copy()
smoothed_data['Y'] = smoothed_data['Y'].rolling(window=20, min_periods=1).mean()

# 크랙 위치
crack_start = 40.211
crack_end = 106.489

crack_mid = (crack_end+crack_start)/2

# 데이터 시각화
plt.figure(figsize=(10, 6))

# Normal 구간
plt.plot(smoothed_data[(smoothed_data['X'] < crack_start) | (smoothed_data['X'] > crack_end)]['X'], 
         smoothed_data[(smoothed_data['X'] < crack_start) | (smoothed_data['X'] > crack_end)]['Y'], 
         marker='o', linestyle='-', color='b', label='Normal')

# Defect 구간
plt.plot(smoothed_data[(smoothed_data['X'] >= crack_start) & (smoothed_data['X'] <= crack_end)]['X'], 
         smoothed_data[(smoothed_data['X'] >= crack_start) & (smoothed_data['X'] <= crack_end)]['Y'], 
         marker='o', linestyle='-', color='r', label='Defect')

plt.plot(smoothed_data[(smoothed_data['X'] == crack_mid)]['X'],
         smoothed_data[(smoothed_data['X'] == crack_mid)]['Y'],
         marker='o', linestyle='-', color='g', label='Mid')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Plot of Y vs X with Defect Highlighted')
plt.legend()
plt.grid(True)
plt.show()
