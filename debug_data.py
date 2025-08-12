#!/usr/bin/env python3
import sys
sys.path.append('src/horizon_detection')
from data_loader import HorizonDataset
import matplotlib.pyplot as plt
import numpy as np

# 测试数据加载器
dataset = HorizonDataset(
    images_dir="data/raw",
    annotations_dir="data/horizon",
    image_size=(512, 512),
    augment=False,
    extract_features=False
)

print(f"Dataset size: {len(dataset)}")

# 取第一个样本
sample = dataset[0]
print(f"Sample keys: {sample.keys()}")
print(f"Image shape: {sample['image'].shape}")
print(f"Target mask shape: {sample['target_mask'].shape}")
print(f"Target mask unique values: {np.unique(sample['target_mask'])}")
print(f"Target mask sum: {np.sum(sample['target_mask'])}")

# 可视化
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 原图
axes[0].imshow(sample['image'])
axes[0].set_title('Original Image')
axes[0].axis('off')

# Horizon mask
axes[1].imshow(sample['horizon_mask'], cmap='gray')
axes[1].set_title(f'Horizon Mask (sum={np.sum(sample["horizon_mask"])})')
axes[1].axis('off')

# Target mask  
axes[2].imshow(sample['target_mask'], cmap='gray')
axes[2].set_title(f'Target Mask (sum={np.sum(sample["target_mask"])})')
axes[2].axis('off')

plt.tight_layout()
plt.savefig('debug_data_loading.png', dpi=150, bbox_inches='tight')
print("Debug image saved as debug_data_loading.png")