import os
import cv2
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import colorsys
from PIL import Image, ImageDraw, ImageFont

# 假设这些是你的自定义模块
from dataloderv2_norm_fog_ll import FusionDataset
from models.net import Net
from models.common import clamp
from models.utils_bbox import postprocess, decode_bbox

import argparse


def get_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='Test Fusion Network and Save Results')

    # --- 路径参数 ---
    parser.add_argument('--weights_path', type=str, default='', help='训练好的权重路径')
    parser.add_argument('--vis_p', type=str, default='', help='测试集可见光路径')
    parser.add_argument('--inf_p', type=str, default='', help='测试集红外路径')
    parser.add_argument('--label_p', type=str, default='', help='标签路径')
    parser.add_argument('--save_fusion_dir', type=str, default='', help='保存融合图像的目录')

    # --- 超参数 ---
    parser.add_argument('--batch_size', type=int, default=1, help='批大小')
    parser.add_argument('--input_shape', type=tuple, default=(640, 640), help='输入图像尺寸 (宽, 高)')
    parser.add_argument('--num_classes', type=int, default=6, help='类别数')

    # 注意：nms_iou 在原代码中定义了但未使用，这里依然保留
    parser.add_argument('--nms_iou', type=float, default=0.7, help='NMS IOU 阈值')

    return parser.parse_args()


def main():
    # 1. 获取参数
    args = get_args()

    os.makedirs(args.save_fusion_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 3. 加载数据集
    # 注意：原代码中 vis_gt_path 等参数直接复用了 vis_p，这里保持一致
    test_dataset = FusionDataset(
        vis_path=args.vis_p,
        inf_path=args.inf_p,
        vis_gt_path=args.vis_p,
        vis_ll_path=args.vis_p,
        vis_fog_path=args.vis_p,
        inf_gt_path=args.inf_p,
        label_path=args.label_p,
        input_shape=args.input_shape,
        num_classes=args.num_classes,
        train=False
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # 4. 加载模型
    model = Net().to(device)
    if os.path.exists(args.weights_path):
        # 建议添加 map_location 以防止设备不匹配报错
        model.load_state_dict(torch.load(args.weights_path, map_location=device))
        print(f"Loaded weights from {args.weights_path}")
    else:
        print(f"Weights not found at {args.weights_path}")
        return

    model.eval()

    print("Start Testing...")
    with torch.no_grad():
        for i, (vis, inf, img_names) in enumerate(tqdm(test_loader)):
            # 假设 img_names 是列表或元组
            current_name = os.path.splitext(img_names[0])[0]

            vis = vis.to(device)
            inf = inf.to(device)

            # 前向传播
            fused = model(vis, inf)
            fused = clamp(fused)

            # --- 1. 保存融合图像 ---
            # squeeze(0) 去掉 batch 维度
            fused_img = fused.squeeze(0).cpu().numpy()
            # (C, H, W) -> (H, W, C)
            fused_img = np.transpose(fused_img, (1, 2, 0))
            fused_img = (fused_img * 255).astype(np.uint8)

            pil_img = Image.fromarray(fused_img, mode='RGB')
            save_path = os.path.join(args.save_fusion_dir, f"{current_name}.png")
            pil_img.save(save_path)

    print("Testing Finished.")


if __name__ == '__main__':
    main()