import torch
import torch.nn as nn
import torch.nn.functional as F


# 特征转化为节点
class FeatureToNode(nn.Module):
    def __init__(self, in_channels, out_dim):
        super().__init__()
        self.proj = nn.Linear(in_channels, out_dim)

    def forward(self, x):
        """
        x: [B, C, H, W]
        """
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)  # [B, C]
        x = self.proj(x)  # [B, D]
        return x

# 来自节点的特征，转回原本的维度
class NodeToFeature(nn.Module):
    def __init__(self, in_channels, out_dim):
        super().__init__()
        self.proj = nn.Linear(in_channels, out_dim)

    def forward(self, x):
        return self.proj(x)




# 稀疏注意力，将弱关系边直接删除 [2.5, 1.2, 0.3, -0.5]->[0.75, 0.25, 0, 0].
# 整体的过程就是
# Step1: 输入归一化，得到z
# step2: 对z排序，得到z_sort
# step3: 定义 k = [1,2, ..., N]
# step4：z_c = [z1, z1+z1, ...]
# step5: e_i = max(0, zi-tau)，选取强关系的边保存，并输出

class Sparsemax(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        # 1. 增加输入检查：如果输入本身有 NaN，直接返回或处理，防止污染后续计算
        if torch.isnan(input).any():
            # 这里可以选择抛出异常或者返回一个全 0 张量，视你的容错需求而定
            # 为了训练稳定性，建议返回全 0 或均值
            return torch.zeros_like(input)

        # 2. 减去最大值，防止指数爆炸 (标准操作)
        z = input - input.max(dim=self.dim, keepdim=True)[0]

        # 3. 排序
        z_sorted, _ = torch.sort(z, descending=True, dim=self.dim)

        # 4. 计算 k 和 cumsum
        # 注意：确保 k 的数据类型与 input 一致，防止类型不匹配
        k = torch.arange(1, z.size(self.dim) + 1, device=input.device, dtype=input.dtype)
        # 为了广播机制，需要调整 k 的形状以匹配 self.dim
        # 这是一个通用的广播形状构造方法
        shape = [1] * input.dim()
        shape[self.dim] = -1
        k = k.view(shape)

        z_cumsum = torch.cumsum(z_sorted, dim=self.dim)

        # 5. 计算 support (判断哪些维度被激活)
        # 公式：1 + k * z_sorted > z_cumsum
        support = (1 + k * z_sorted > z_cumsum)

        # 6. 计算 k_z (激活维度的数量)
        k_z = support.sum(dim=self.dim, keepdim=True)

        # 7. 防止除以 0 的核心修改
        # 强制 k_z 至少为 1，防止 gather 索引为 -1，同时也防止分母为 0
        k_z_safe = torch.clamp(k_z, min=1)

        # 8. 计算 tau
        # 使用 k_z_safe 进行 gather，确保索引合法
        # 注意：gather 的索引必须是 Long 类型
        tau = (z_cumsum.gather(self.dim, k_z_safe - 1) - 1) / k_z_safe

        # 9. 最终输出
        output = torch.clamp(z - tau, min=0)

        # 10. 二次检查：如果输出中有 NaN（极少见，但可能发生），强制置 0
        if torch.isnan(output).any():
            return torch.zeros_like(input)

        return output

# 自感知关系层
# 先学习得到不同node之间的关系，再进行稀疏裁剪 #
class RelationSelfAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(2 * dim, 1)
        self.act = nn.ReLU()
        # self.sparsemax = Sparsemax(dim=-1)

    def forward(self, x):
        """
        x: [B, N, D]  (N=8)，
        """
        B, N, D = x.shape

        xi = x.unsqueeze(2).repeat(1, 1, N, 1)
        xj = x.unsqueeze(1).repeat(1, N, 1, 1)

        x_cat = torch.cat([xi, xj], dim=-1)

        r = self.fc(x_cat).squeeze(-1)
        r = self.act(r)

        adj = r

        return adj

# CheConv, {PyG} 2.0: Scalable Learning on Real World Graphs
class ChebConv(nn.Module):
    def __init__(self, in_channels, out_channels, K, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K

        # 每一阶都有独立参数
        self.weight = nn.Parameter(
            torch.Tensor(K, in_channels, out_channels)
        )

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        """
        x:   [B, N, F]
        adj: [B, N, N]
        """

        B, N, _ = x.shape
        device = x.device

        # ---------- 1. 构造 Laplacian ----------
        I = torch.eye(N, device=device).unsqueeze(0).expand(B, -1, -1)

        deg = adj.sum(dim=-1)  # [B, N]
        D = torch.diag_embed(deg)

        L = D - adj  # unnormalized Laplacian

        # ---------- 2. 归一化 ----------
        deg_inv_sqrt = torch.pow(deg + 1e-6, -0.5)
        D_inv_sqrt = torch.diag_embed(deg_inv_sqrt)

        L_norm = D_inv_sqrt @ L @ D_inv_sqrt  # L_sym

        # ---------- 3. 计算 λ_max ----------
        # 严格来说要用最大特征值，这里用近似（推荐）
        lambda_max = 2.0

        # ---------- 4. rescale ----------
        L_tilde = (2.0 / lambda_max) * L_norm - I

        # ---------- 5. Chebyshev 多项式 ----------
        Tx = [x]  # T0(x)

        if self.K > 1:
            Tx_1 = torch.bmm(L_tilde, x)
            Tx.append(Tx_1)

        for k in range(2, self.K):
            Tx_k = 2 * torch.bmm(L_tilde, Tx[-1]) - Tx[-2]
            Tx.append(Tx_k)

        # ---------- 6. 聚合 ----------
        out = 0
        for k in range(self.K):
            out = out + torch.matmul(Tx[k], self.weight[k])

        # ---------- 7. bias ----------
        if self.bias is not None:
            out = out + self.bias

        return out


# 对于特有特征进行融合
from .wt_conv_fusion import WTConv2d_VIF

class MultiModalGraphFusion(nn.Module):
    def __init__(self, out_dim=128, layer_dims=[64, 128, 320, 512]):
        super().__init__()
        self.fusion_modules = nn.ModuleList()

        for dim in layer_dims:
            self.fusion_modules.append(WTConv2d_VIF(dim, dim))

    def forward(self, vi_features, ir_features):
        fused_features = []
        for i, fusion_layer in enumerate(self.fusion_modules):
            fused = fusion_layer(vi_features[i], ir_features[i])
            fused_features.append(fused)
        return fused_features

class baseFusion(nn.Module):
    def __init__(self, layer_dims=[64, 128, 320, 512]):
        super().__init__()
        self.conv = nn.ModuleList()
        for dim in layer_dims:
            self.conv.append(nn.Conv2d(dim*2, dim, stride=1, padding=0, kernel_size=1))

    def forward(self, vi_feature, ir_feature):
        fused_features = []
        idx = 0
        for i, j in zip(vi_feature, ir_feature):
            fused_features.append(self.conv[idx](torch.cat([i,j],dim=1)))
            idx = idx+1

        return fused_features


if __name__ == '__main__':
    x1 = torch.randn(1, 16, 640, 480).cuda()
    x2 = torch.randn(1, 32, 320, 240).cuda()
    x3 = torch.randn(1, 64, 160, 120).cuda()
    x4 = torch.randn(1, 128, 80, 60).cuda()
    feates = [x1, x2, x3, x4, x1, x2, x3, x4]
    model = MultiModalGraphFusion(out_dim=64, layer1_dim=16, layer2_dim=32, layer3_dim=64, layer4_dim=128).cuda()
    for xx in model(feates, feates):
        print(xx.shape)
    # print(model(feates))

