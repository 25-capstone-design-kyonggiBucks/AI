import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.autograd import Variable
import argparse
import os

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
    "air0.6m": "Void",
    "air1m": "Void",
    "air1.4m": "Void",
    "sand0%": "Soil",
    "sand18%": "Soil",
    "sand_weathered": "Soil",
    "normal": "Normal",
    "water": "Water"
}

# ArgumentParser 객체 생성
parser = argparse.ArgumentParser(description="Process a directory path.")
parser.add_argument('model_path', type=str, help='Path to the model file')
parser.add_argument('dir_path', type=str, help='Path to the directory containing CSV files')

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

# 디렉토리 내의 모든 CSV 파일 처리
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

for filename in os.listdir(args.dir_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(args.dir_path, filename)
        data = pd.read_csv(file_path, header=None, names=['X', 'Y'])
        reflectance = data['Y'].values  # 첫 번째 열을 제외한 두 번째 열 선택

        # 데이터를 PyTorch 텐서로 변환
        inputs = torch.tensor(reflectance, dtype=torch.float32).unsqueeze(0).unsqueeze(2).to(device)

        # 모델로 예측 수행
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)

        # 예측 결과에 따라 함수 호출
        for pred in predicted:
            item = class_names[pred.item()]
            ab = abnormal[item]
            print(f"{file_path}의 예측된 클래스: {ab}")
