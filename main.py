# main.py
from train import train_model
from inference import inference

if __name__ == "__main__":
    # 1. 모델 학습
    train_model()

    # 2. 추론
    rgb_img_path = "C:/JeongEunYang/dataset/rgb/test_image.jpg"
    depth_img_path = "C:/JeongEunYang/dataset/depth/test_image.npy"
    inference(rgb_img_path, depth_img_path)
