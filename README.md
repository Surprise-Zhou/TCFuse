# TCFuse

RGB-T（可见光–红外）图像融合与目标检测网络。

双流 MiT（SegFormer）骨干分别提取可见光与红外特征，将特征解耦为**基础分量**与**细节分量**，
分别用卷积融合与小波/图融合模块处理，再经 MLP 解码头重建融合图像，
同时由 CenterNet 风格的检测头输出目标检测结果。

## 目录结构

```
TCFuse/
├── testv3.py                 # 推理 / 测试入口
└── models/
    ├── net.py                # 主网络 Net：双流骨干 + 融合 + 解码 + 检测头
    ├── backbone.py           # MiT (mit_b0 ~ mit_b5) 骨干与特征解耦模块
    ├── grahp_fusion.py       # 多模态融合模块（WTConv 融合 / 图卷积组件）
    ├── wt_conv_fusion.py     # 小波卷积融合 WTConv2d_VIF
    ├── wavelet.py            # 小波变换基础算子
    ├── resnet50.py           # CenterNet 解码器与检测头
    ├── fused_all_features_for_detect.py  # 多层特征聚合后送检测分支
    ├── utils_bbox.py         # 检测后处理、解码与 NMS
    ├── fusion_loss.py        # 融合损失
    ├── loss_con.py           # 对比 / 一致性损失
    ├── loss_detect.py        # 检测损失
    ├── common.py             # 公共工具（如 clamp）
    ├── net_wo_moe.py         # 消融：去掉 MoE 的版本
    └── net_wo_fusion_module.py  # 消融：去掉融合模块的版本
```

## 环境依赖

- Python 3.8+
- PyTorch、torchvision
- opencv-python、numpy、Pillow、tqdm、PyWavelets

```bash
pip install torch torchvision opencv-python numpy Pillow tqdm PyWavelets
```

## 推理

权重文件（`uav_final/with_det.pth`）因体积超过 GitHub 单文件限制未纳入仓库，
请在本地准备好后按下面的方式运行：

```bash
python testv3.py \
    --weights_path uav_final/with_det.pth \
    --vis_p  <可见光测试集路径> \
    --inf_p  <红外测试集路径> \
    --label_p <标签路径> \
    --save_fusion_dir <融合结果保存目录> \
    --input_shape "(640, 640)" \
    --num_classes 6 \
    --batch_size 1
```

融合结果会以 PNG 保存到 `--save_fusion_dir`。

## 模型要点

- **双流骨干**：可见光 / 红外各一条 MiT 分支（`phi` 可选 `b0`~`b5`），支持 MoE（`num_experts`）。
- **特征解耦**：`get_feature_depart_b0/b1` 把每层特征拆成基础分量 `base` 与细节分量 `de`。
- **分层融合**：基础分量经 1×1 卷积融合（`baseFusion`），细节分量经小波卷积融合（`MultiModalGraphFusion`），
  再用 `concat_fusion` 合并两路结果。
- **融合解码**：`FusionHead` 对四层特征做 MLP 对齐与拼接，经 `conv_out` 输出 3 通道融合图。
- **检测分支**：`fused_all_features_for_detect` 聚合四层融合特征，送入 `resnet50_Decoder` + `resnet50_Head`，
  输出类别热图与边界框回归。
- **消融变体**：`net_wo_moe.py`、`net_wo_fusion_module.py` 用于对比实验。

## 备注

`testv3.py` 依赖数据加载模块 `dataloderv2_norm_fog_ll`（`FusionDataset`），
该模块不在此目录中，需自行提供。

## License

部分代码改编自 NVIDIA SegFormer（见 `models/net.py` 头部版权声明）。
