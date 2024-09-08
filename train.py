# train.py
import torch
from torch.utils.data import DataLoader
from dataset import RGBD_Dataset
from model import YOLOSegmentModel
from params import Config

def train_model():
    # 데이터셋 준비
    dataset = RGBD_Dataset(f"{Config.dataset_path}/rgb", f"{Config.dataset_path}/depth")
    dataloader = DataLoader(dataset, batch_size=Config.batch_size, shuffle=True)
    
    # 모델 준비
    model = YOLOSegmentModel(Config.yolo_model)

    # 손실 함수 및 옵티마이저
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.learning_rate)

    # 학습 루프
    for epoch in range(Config.epochs):
        for images, _, labels in dataloader:  # RGB 이미지만 사용
            optimizer.zero_grad()
            
            # 예측
            outputs = model.predict(images)

            # 손실 계산 (여기서는 간단한 코드로 예시)
            loss = torch.nn.CrossEntropyLoss()(outputs, labels)

            # 역전파 및 옵티마이저 스텝
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{Config.epochs}, Loss: {loss.item()}")

    # 모델 저장
    torch.save(model.model.state_dict(), Config.model_save_path)
