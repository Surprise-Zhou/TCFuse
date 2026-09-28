import torch
import torch.nn as nn
import torch.nn.functional as F


# class EntropyBasedTripletLoss(nn.Module):
#     def __init__(self, alpha=1.0, beta=1.0, eps=1e-8):
#         """
#         基于信息熵的三元组损失 (优化版)
#         """
#         super().__init__()
#         self.alpha = alpha
#         self.beta = beta
#         self.eps = eps
#
#     def _calculate_entropy(self, x):
#         """
#         计算特征图的微分熵 H(x) = 0.5 * ln(2 * pi * e * variance)
#         """
#         # 1. 展平特征 (B, C*H*W)
#         batch_size = x.shape[0]
#         x_flat = x.view(batch_size, -1)
#
#         # 2. 计算方差
#         # 使用无偏估计 (ddof=1)，在样本数较少时更稳定
#         variance = x_flat.var(dim=1, unbiased=True)
#
#         # 3. 计算熵
#         # 使用 torch.lerp 或 clamp 确保方差非负，增加鲁棒性
#         variance = torch.clamp(variance, min=self.eps)
#         entropy = 0.5 * torch.log(2 * torch.pi * torch.e * variance)
#
#         return entropy
#
#     def forward(self, vi_base, ir_base, vi_detail, ir_detail):
#         """
#         计算总损失 (完全向量化版本)
#         """
#         b, c, h, w = vi_base.shape
#
#         # ==========================================
#         # Loss 1: KL 散度比率损失
#         # ==========================================
#         # 直接使用 view 而不是重复的变量赋值
#         vi_base_flat = vi_base.view(b, -1)
#         ir_base_flat = ir_base.view(b, -1)
#         vi_detail_flat = vi_detail.view(b, -1)
#         ir_detail_flat = ir_detail.view(b, -1)
#
#         # KL散度计算
#         # 使用 F.softmax 的输出作为目标分布通常更稳定
#         p_base = F.softmax(ir_base_flat, dim=1)
#         log_q_base = F.log_softmax(vi_base_flat, dim=1)
#         kl_base = F.kl_div(log_q_base, p_base, reduction='batchmean')
#
#         p_detail = F.softmax(ir_detail_flat, dim=1)
#         log_q_detail = F.log_softmax(vi_detail_flat, dim=1)
#         kl_detail = F.kl_div(log_q_detail, p_detail, reduction='batchmean')
#
#         # 防止除零
#         loss1 = kl_base / (kl_detail + self.eps)
#
#         # ==========================================
#         # Loss 2: 基于信息熵的三元组损失 (向量化)
#         # ==========================================
#
#         # 1. 构建三元组样本
#         anchor_vi = vi_base
#         anchor_ir = ir_base
#         positive = vi_base + ir_base
#         negative = vi_base + vi_detail
#
#         # 2. 计算各样本的熵值 (B,)
#         h_anchor_vi = self._calculate_entropy(anchor_vi)
#         h_anchor_ir = self._calculate_entropy(anchor_ir)
#         h_positive = self._calculate_entropy(positive)
#         h_negative = self._calculate_entropy(negative)
#
#         # 3. 计算熵距离 (B,)
#         # 使用 L1 距离
#         dist_pos_vi = torch.abs(h_anchor_vi - h_positive)
#         dist_neg_vi = torch.abs(h_anchor_vi - h_negative)
#
#         dist_pos_ir = torch.abs(h_anchor_ir - h_positive)
#         dist_neg_ir = torch.abs(h_anchor_ir - h_negative)
#
#         # 4. 构建损失
#         # 目标：最小化正样本熵距离，最大化负样本熵距离
#         # 我们使用 ReLU 形式的 triplet loss，即 max(0, dist_pos - dist_neg + margin)
#         # 这里我们引入一个小的 margin，鼓励模型做得更好，而不仅仅是满足条件
#         margin = 0.1
#         loss2_vi = torch.mean(torch.clamp(dist_pos_vi - dist_neg_vi + margin, min=0))
#         loss2_ir = torch.mean(torch.clamp(dist_pos_ir - dist_neg_ir + margin, min=0))
#
#         loss2 = loss2_vi + loss2_ir
#
#         # ==========================================
#         # 总损失
#         # ==========================================
#         total_loss = self.alpha * loss1 + self.beta * loss2
#
#         return total_loss

class EntropyBasedTripletLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, margin=0.5):
        """
        高效版：基于方差的特征约束 + 标准三元组损失
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.margin = margin

    def _calculate_variance_loss(self, x, target_variance=None):
        """
        计算特征图的方差损失
        目标：鼓励特征图具有足够的信息量（即方差不能太小，也不能太大）
        """
        # 直接在 (B, C, H, W) 上计算方差，不展平
        # dim=[1,2,3] 表示计算每个样本在整个空间通道上的方差
        variance = x.var(dim=[1, 2, 3], unbiased=True)

        # 简单的约束：让方差接近 1.0 (标准化分布)
        # 这比计算复杂的微分熵公式要快得多，且能防止特征坍塌
        if target_variance is None:
            target_variance = torch.ones_like(variance)

        return F.mse_loss(variance, target_variance)

    def forward(self, vi_base, ir_base, vi_detail, ir_detail):
        """
        简化后的前向传播
        """
        # ==========================================
        # Loss 1: 特征分布一致性 (替代 KL 散度)
        # ==========================================
        # 目标：让可见光基底的统计特性接近红外基底
        # 使用均值和方差的 MSE 替代昂贵的 KL 散度
        loss_dist = F.mse_loss(vi_base, ir_base)

        # ==========================================
        # Loss 2: 高效三元组损失
        # ==========================================
        # 这里的逻辑简化为：

        # 使用 PyTorch 原生的 triplet loss，这是 CUDA 优化过的
        # 注意：这里直接对特征图计算，PyTorch 会自动处理维度
        loss_triplet = F.triplet_margin_loss(
            vi_base,  # Anchor
            ir_base+vi_base,  # Positive
            vi_detail,  # Negative
            margin=self.margin
        )

        # 可选：增加一个细节损失的约束，防止细节图包含过多基底信息
        # loss_detail_reg = torch.mean(torch.abs(vi_detail * ir_base))

        total_loss = self.alpha * loss_dist + self.beta * loss_triplet

        return total_loss


# --- 测试代码 ---
if __name__ == "__main__":
    batch_size = 4
    channels = 64
    height = 32
    width = 32

    vi_base = torch.randn(batch_size, channels, height, width) * 0.5
    ir_base = torch.randn(batch_size, channels, height, width) * 0.5
    vi_detail = torch.randn(batch_size, channels, height, width) * 2.0
    ir_detail = torch.randn(batch_size, channels, height, width) * 2.0

    criterion = EntropyBasedTripletLoss(alpha=1.0, beta=0.5)

    total_loss = criterion(vi_base, ir_base, vi_detail, ir_detail)

    print(f"Total Loss: {total_loss.item():.4f}")