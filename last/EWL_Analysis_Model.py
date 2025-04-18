import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.autograd import Variable
import argparse

# 0 -> air06
# 1 -> air1
# 2 -> air14
# 3 -> normal
# 4 -> sand0
# 5 -> sand18
# 6 -> sand_weat
# 7 -> water

# 클래스 이름을 저장하는 리스트 (임시로 주석 기준으로 설정)
class_names = ["air0.6m", "air1m", "air1.4m", "normal", "sand0%", "sand18%", "sand_weathered", "water"]
abnormal = {
    "air0.6m" : "Void",
    "air1m" : "Void",
    "air1.4m" : "Void",
    "sand0%" : "Soil",
    "sand18%" : "Soil",
    "sand_weathered" : "Soil",
    "normal" : "Normal",
    "water" : "Water"
}

# ArgumentParser 객체 생성
parser = argparse.ArgumentParser(description="Process a file path.")
parser.add_argument('model_path', type=str, help='Path to the file')
parser.add_argument('file_path', type=str, help='Path to the file')

# 명령줄 인자 파싱
args = parser.parse_args()

# 파이프라인 사용 예시
data_folder = 'data'  # CSV 파일이 위치한 폴더 경로
window_size = 5  # 이동 평균에 사용할 윈도우 크기

num_classes = 8
input_size = 1
hidden_size = 256
num_layers = 8
dropout = 0.1

# 클래스별 임계값 딕셔너리 (예시)
crack_threshold = {
    "air1m": {"left": 1, "right": 0.26},
    "air0.6m": {"left": -0.12, "right": 0.15},
    "air1.4m": {"left": -0.145, "right": 0.17},
    "normal": {"left": -0.11, "right": 0.175},
    "sand_weathered": {"left": -0.11, "right": 0.165},
    "sand0%": {"left": -0.1, "right": 0.17},
    "sand18%": {"left": -0.11, "right": 0.164},
    "water": {"left": -0.1, "right": 0.14},
}

def calculate_crack_size(data, window_size=5, item=None):
    if item not in crack_threshold:
        raise ValueError(f"알 수 없는 클래스: {item}")

    # 해당 클래스의 임계값 가져오기
    left_threshold = crack_threshold[item]["left"]
    right_threshold = crack_threshold[item]["right"]

    # 각 열에 대해 이동 평균을 계산하여 smoothing
    smoothed_data = data.copy()
    smoothed_data['Y'] = smoothed_data['Y'].rolling(window=window_size, min_periods=1).mean()

    # 첫 번째 열을 x축(시간), 두 번째 열을 y축으로 설정
    x = data['X'].values
    y = data['Y'].values

    # x축의 센터포인트 찾기
    center_index = len(x) // 2

    # 중간 앞쪽 구간에서 4초 동안 Y 값의 변화량이 가장 큰 곳 찾기 (conc_start)
    left_half_data = smoothed_data.iloc[:center_index]
    max_y_change = 0  # 최대 Y 변화량을 저장할 변수
    conc_start = left_half_data.iloc[0]['X']  # 초기 시작점을 첫 번째 X로 설정

    for i in range(len(left_half_data) - 1):
        x_start = left_half_data.iloc[i]['X']
        x_end = x_start + 4  # 4초 후의 X 값을 계산

        if x_end > left_half_data.iloc[-1]['X']:
            break

        data_in_range = left_half_data[(left_half_data['X'] >= x_start) & (left_half_data['X'] <= x_end)]
        
        if len(data_in_range) > 1:
            y_change = data_in_range['Y'].max() - data_in_range['Y'].min()  # Y 값의 최대-최소 차이

            if y_change > max_y_change:
                max_y_change = y_change
                conc_start = x_start + 4

    # 중간 이후 구간에서 4초 동안 Y 값의 변화량이 가장 큰 곳 찾기 (conc_end)
    right_half_data = smoothed_data.iloc[center_index:]
    max_y_change = 0
    conc_end = right_half_data.iloc[-1]['X']

    for i in range(len(right_half_data) - 1):
        x_start = right_half_data.iloc[i]['X']
        x_end = x_start + 4

        if x_end > right_half_data.iloc[-1]['X']:
            break

        data_in_range = right_half_data[(right_half_data['X'] >= x_start) & (right_half_data['X'] <= x_end)]
        
        if len(data_in_range) > 1:
            y_change = data_in_range['Y'].max() - data_in_range['Y'].min()

            if y_change > max_y_change:
                max_y_change = y_change
                conc_end = x_start + 4

    # Crack Start: conc_start + 10 이후부터 탐색
    start_search_data = smoothed_data[smoothed_data['X'] > (conc_start + 10)]
    gradients_after_conc_start = np.gradient(start_search_data['Y'].values, start_search_data['X'].values)
    left_threshold_index = np.where(gradients_after_conc_start <= left_threshold)[0]
    crack_start = start_search_data.iloc[left_threshold_index[0]]['X'] if len(left_threshold_index) > 0 else conc_start + 10

    # Crack End: conc_end - 5부터 conc_start까지 역방향으로 탐색
    end_search_data = smoothed_data[(smoothed_data['X'] >= conc_start) & (smoothed_data['X'] <= (conc_end - 5))]
    gradients_before_conc_end = np.gradient(end_search_data['Y'].values[::-1], end_search_data['X'].values[::-1])
    right_threshold_index = np.where(gradients_before_conc_end >= right_threshold)[0]
    crack_end = end_search_data.iloc[-right_threshold_index[0]]['X'] if len(right_threshold_index) > 0 else conc_end - 5

    return conc_start, conc_end - 3, crack_start, crack_end

class LSTM(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, dropout=0.5):
        super(LSTM, self).__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size

        # LSTM layer
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=dropout)

        # Fully connected layers and Batch normalization
        self.fc_1 = nn.Linear(hidden_size, 128)
        self.bn_1 = nn.BatchNorm1d(128)
        self.dropout_1 = nn.Dropout(dropout)
        self.fc = nn.Linear(128, num_classes)
        self.bn_2 = nn.BatchNorm1d(num_classes)
        self.dropout_2 = nn.Dropout(dropout)

        # Activation function
        self.gelu = nn.GELU()

    def forward(self, x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(x.device)
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(x.device)

        # LSTM output
        output, (hn, cn) = self.lstm(x, (h_0, c_0))
        hn = hn[-1]

        # Fully connected layer 1
        out = self.fc_1(hn)
        out = self.bn_1(out)
        out = self.gelu(out)
        out = self.dropout_1(out)

        # Fully connected layer 2
        out = self.fc(out)
        out = self.bn_2(out)
        out = self.dropout_2(out)

        return out

# LSTM 모델 로드
model = LSTM(num_classes, input_size, hidden_size, num_layers, dropout)
model.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))
model.eval()

# CSV 파일 로드        
data = pd.read_csv(args.file_path, header=None, names=['X', 'Y'])
reflectance = data['Y'].values  # 첫 번째 열을 제외한 두 번째 열 선택

# 데이터를 PyTorch 텐서로 변환
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
inputs = torch.tensor(reflectance, dtype=torch.float32).unsqueeze(0).unsqueeze(2).to(device)
model.to(device)

# 모델로 예측 수행
outputs = model(inputs)
_, predicted = torch.max(outputs, 1)

# 예측 결과에 따라 함수 호출
for pred in predicted:
    item = class_names[pred.item()]
    ab = abnormal[item]
    print(f"{args.file_path}의 예측된 클래스: {ab}")

   # 정상 클래스가 아닌 경우에만 크랙 감지 동작
    if item != "normal":
        try:
            conc_start, conc_end, crack_start, crack_end = calculate_crack_size(data, item=item)
            print(f"{args.file_path}의 콘크리트 구조물 감지 - 시작 시간: {conc_start:.3f}, 종료 시간: {conc_end:.3f}")
            print(f"{args.file_path}의 결함 감지 - 시작 시간: {crack_start:.3f}, 종료 시간: {crack_end:.3f}")
            print()
        except ValueError as e:
            print(f"{args.file_path}에서 오류 발생: {e}")
