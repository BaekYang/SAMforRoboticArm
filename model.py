# model.py
import torch
from ultralytics import YOLO

class YOLOSegmentModel:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def predict(self, img):
        results = self.model(img)  # YOLOv8을 이용한 예측
        return results

    def get_xyz(self, depth_image, bounding_box):
        """
        depth_image: depth 카메라로 얻은 깊이 정보
        bounding_box: YOLO 모델의 탐지 결과에서 얻은 bounding box 좌표
        
        bounding_box에서의 중간 지점의 depth 값을 이용해 XYZ 좌표를 추정
        """
        x_center = int((bounding_box[0] + bounding_box[2]) / 2)
        y_center = int((bounding_box[1] + bounding_box[3]) / 2)
        z_value = depth_image[y_center, x_center]  # 중간 지점에서의 깊이 값

        # XYZ 좌표 계산 (여기서는 예시적으로 변환, 실제 변환식은 카메라의 파라미터에 의존)
        x = (x_center - depth_image.shape[1] / 2) * z_value / 1000  # 예시 변환
        y = (y_center - depth_image.shape[0] / 2) * z_value / 1000
        z = z_value

        return x, y, z
