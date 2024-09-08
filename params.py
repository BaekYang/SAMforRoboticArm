# params.py

class Config:
    # 모델 설정
    num_classes = 3  # 접시(dish), 주전자(pot), 컵(cup)
    labels = {0: 'dish', 1: 'pot', 2: 'cup'}

    # 경로 설정
    base_dir = 'C:/JeongEunYang'
    dataset_path = f'{base_dir}/dataset/'  # 데이터셋 경로
    model_save_path = f'{base_dir}/models/yolo_segment_model.pt'  # 모델 저장 경로

    # 학습 설정
    epochs = 50
    batch_size = 16
    learning_rate = 0.001

    # YOLOv8 모델 설정
    yolo_model = 'yolov8n-seg.pt'  # 사용할 YOLOv8 세그먼트 모델
    input_size = 640  # 이미지 크기
