# dataset.py
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class RGBD_Dataset(Dataset):
    def __init__(self, rgb_dir, depth_dir, transform=None):
        self.rgb_dir = rgb_dir
        self.depth_dir = depth_dir
        self.rgb_files = os.listdir(rgb_dir)
        self.depth_files = os.listdir(depth_dir)
        self.transform = transform

    def __len__(self):
        return len(self.rgb_files)

    def __getitem__(self, idx):
        # RGB 이미지 로드
        rgb_path = os.path.join(self.rgb_dir, self.rgb_files[idx])
        rgb_image = Image.open(rgb_path).convert('RGB')

        # Depth 이미지 로드
        depth_path = os.path.join(self.depth_dir, self.depth_files[idx])
        depth_image = np.load(depth_path)  # Depth 이미지를 npy로 저장했다고 가정

        if self.transform:
            rgb_image = self.transform(rgb_image)

        return rgb_image, depth_image, 0  # 샘플 레이블 (추후 실제 데이터로 교체)
