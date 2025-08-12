import torch
from horizon_segmentation_model import LightweightUNet, SoilHorizonDepthDataset, get_transforms
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import json
from PIL import Image

def draw_horizon_lines_with_labels(ax, y_list, cm_list, color, label):
    for i, (y, cm) in enumerate(zip(y_list, cm_list)):
        ax.axhline(y, color=color, linestyle='--', linewidth=2, label=label if i == 0 else None)
        ax.text(10, y, f"{cm:.0f} cm", color=color, fontsize=10, va='bottom', ha='left', bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

def main():
    data_dir = Path("data/processed")
    checkpoint_path = Path("checkpoints/best_model.pth")
    save_dir = Path("evaluation_results/compare_predictions")
    save_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LightweightUNet(max_horizons=7)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)

    dataset = SoilHorizonDepthDataset(str(data_dir), transform=get_transforms('val'))

    for sample in dataset:
        sample_name = sample['sample_name']
        processed_img_path = data_dir / f"{sample_name}_processed.png"
        original_img_path = Path("data/raw") / f"{sample_name}.jpg"
        if not original_img_path.exists():
            original_img_path = Path("data/raw") / f"{sample_name}.png"
        metadata_path = data_dir / f"{sample_name}_metadata.json"

        # 使用原始图片而不是处理过的图片，这样尺子刻度可见
        if original_img_path.exists():
            img = np.array(Image.open(original_img_path))
        else:
            # 如果找不到原始图片，使用处理过的图片
            img = np.array(Image.open(processed_img_path))

        # 读取 metadata
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        horizon_info = metadata['horizon_info']
        ruler_info = metadata['ruler_info']

        # 1. 真实分割线（像素位置和厘米标签）
        gt_pixel_lines = [seg[1] for seg in horizon_info['pixel_horizons']]
        gt_cm_labels = [seg[1] for seg in horizon_info['horizon_depths_cm']]

        # 2. 预测分割线（直接用原图比例换算）
        img_for_model = sample['image'].unsqueeze(0).to(device)
        with torch.no_grad():
            pred_depths = model(img_for_model).cpu().numpy()[0]
        valid_num = sample['num_horizons']
        pred_depths = pred_depths[:valid_num]

        top_pixel = ruler_info['line_coords'][1]  # 通常为0
        scale_ratio = ruler_info['scale_ratio']   # 每厘米多少像素（原图）
        pred_pixel_lines = [top_pixel + d * scale_ratio for d in pred_depths]
        pred_cm_labels = [round(d) for d in pred_depths]

        # 可视化
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        axs[0].imshow(img)
        axs[0].set_title("Original")
        axs[0].axis('off')

        axs[1].imshow(img)
        draw_horizon_lines_with_labels(axs[1], gt_pixel_lines, gt_cm_labels, color='g', label='GT')
        axs[1].set_title("Ground Truth Horizons")
        axs[1].axis('off')

        axs[2].imshow(img)
        draw_horizon_lines_with_labels(axs[2], pred_pixel_lines, pred_cm_labels, color='r', label='Pred')
        axs[2].set_title("Predicted Horizons")
        axs[2].axis('off')

        plt.tight_layout()
        plt.savefig(save_dir / f"{sample_name}_compare.png", dpi=200)
        plt.close()

if __name__ == "__main__":
    main()