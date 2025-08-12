#!/usr/bin/env python3
import torch
import numpy as np
import cv2
import sys
sys.path.append('src/horizon_detection')
from horizon_model import create_horizon_model
import torch.nn.functional as F

# 加载模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = create_horizon_model('unet', num_classes=2)
# 加载完整的checkpoint
checkpoint = torch.load('best_unet_model.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

# 加载测试图片
image_path = "data/raw/F57079r.jpg"
image = cv2.imread(image_path)
if image is None:
    print(f"无法加载图像: {image_path}")
    exit()

print(f"Original image shape: {image.shape}")

# 预处理
image_resized = cv2.resize(image, (512, 512))
image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float() / 255.0
image_batch = image_tensor.unsqueeze(0).to(device)

print(f"Input tensor shape: {image_batch.shape}")

# 模型预测
with torch.no_grad():
    output = model(image_batch)
    print(f"Model output shape: {output.shape}")
    print(f"Model output range: {output.min().item():.4f} to {output.max().item():.4f}")
    
    # 应用softmax
    probabilities = F.softmax(output, dim=1)
    print(f"Probabilities shape: {probabilities.shape}")
    print(f"Class 0 prob range: {probabilities[0,0].min().item():.4f} to {probabilities[0,0].max().item():.4f}")
    print(f"Class 1 prob range: {probabilities[0,1].min().item():.4f} to {probabilities[0,1].max().item():.4f}")
    
    # 预测类别 (原始argmax方法)
    pred_mask_argmax = torch.argmax(probabilities, dim=1)
    print(f"ArgMax prediction unique values: {torch.unique(pred_mask_argmax)}")
    print(f"ArgMax Class 1 pixels: {torch.sum(pred_mask_argmax == 1).item()}")
    
    # 使用较低阈值 (0.3)
    class1_probs_tensor = probabilities[0, 1]
    pred_mask_thresh = (class1_probs_tensor > 0.3).long()
    print(f"Threshold 0.3 prediction unique values: {torch.unique(pred_mask_thresh)}")
    print(f"Threshold 0.3 Class 1 pixels: {torch.sum(pred_mask_thresh == 1).item()}")
    
    # 使用更低阈值 (0.2)
    pred_mask_thresh2 = (class1_probs_tensor > 0.2).long()
    print(f"Threshold 0.2 Class 1 pixels: {torch.sum(pred_mask_thresh2 == 1).item()}")
    
    # 检查class 1的概率分布
    class1_probs = probabilities[0, 1].cpu().numpy()
    print(f"Class 1 probability stats:")
    print(f"  Mean: {np.mean(class1_probs):.6f}")
    print(f"  Max: {np.max(class1_probs):.6f}")
    print(f"  Min: {np.min(class1_probs):.6f}")
    print(f"  Pixels > 0.5: {np.sum(class1_probs > 0.5)}")
    print(f"  Pixels > 0.3: {np.sum(class1_probs > 0.3)}")
    print(f"  Pixels > 0.1: {np.sum(class1_probs > 0.1)}")