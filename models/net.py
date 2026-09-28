# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import mit_b0, mit_b1, mit_b2, mit_b3, mit_b4, mit_b5, get_feature_depart_b0, get_feature_depart_b1


class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x
    
class ConvModule(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=0, g=1, act=True):
        super(ConvModule, self).__init__()
        self.conv   = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn     = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act    = nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

class FusionHead(nn.Module):
    """
    moe based head for fusion
    """
    def __init__(self, out_channel=3, in_channels=[32, 64, 160, 256], embedding_dim=768, dropout_ratio=0.1):
        super(FusionHead, self).__init__()
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = in_channels

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(
            c1=embedding_dim*4,
            c2=embedding_dim,
            k=1,
        )

        self.linear_pred    = nn.Conv2d(embedding_dim, out_channel, kernel_size=1)
        self.dropout        = nn.Dropout2d(dropout_ratio)
    
    def forward(self, inputs):
        c1, c2, c3, c4 = inputs

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape
        
        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = F.interpolate(_c4, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))


        x = self.dropout(_c)
        x = self.linear_pred(x)

        return x


from models.grahp_fusion import MultiModalGraphFusion, baseFusion

def add_fusion(list_a, list_b):
    fused_list = []
    for i in range(len(list_a)):
        fused_feat = list_a[i] + list_b[i]
        fused_list.append(fused_feat)
    return fused_list

class concat_fusion(nn.Module):
    def __init__(self, layer_num=[32, 64, 160, 256]):
        super(concat_fusion, self).__init__()
        # self.conv1 = nn.Conv2d(layer_num[0]*2, layer_num[0], kernel_size=1)
        # self.conv2 = nn.Conv2d(layer_num[1]*2, layer_num[1], kernel_size=1)
        # self.conv3 = nn.Conv2d(layer_num[2]*2, layer_num[2], kernel_size=1)
        # self.conv4 = nn.Conv2d(layer_num[3]*2, layer_num[3], kernel_size=1)
        self.conv = nn.ModuleList(
        )
        for i in layer_num:
            self.conv.append(nn.Conv2d(i*2, i, kernel_size=1))

    def forward(self, list_a, list_b):
        fused_list = []
        for i in range(len(list_a)):
            fused_feat = self.conv[i](torch.cat((list_a[i], list_b[i]), dim=1))
            fused_list.append(fused_feat)
        return fused_list



# 导入centernet的decoder和head
from models.resnet50 import resnet50_Decoder, resnet50_Head
from models.wt_conv_fusion import WTConv2d_VIF
from models.fused_all_features_for_detect import fused_all_features_for_detect
import math
class Net(nn.Module):
    def __init__(self, phi = 'b0', pretrained = False, num_classes=1, num_experts=8, depths=None):
        super(Net, self).__init__()
        self.in_channels = {
            'b0': [32, 64, 160, 256], 'b1': [64, 128, 320, 512], 'b2': [64, 128, 320, 512],
            'b3': [64, 128, 320, 512], 'b4': [64, 128, 320, 512], 'b5': [64, 128, 320, 512],
        }[phi]

        self.features_depart = {
            'b0': get_feature_depart_b0(),
            'b1': get_feature_depart_b1(), 'b2': get_feature_depart_b1(),
            'b3': get_feature_depart_b1(), 'b4': get_feature_depart_b1(), 'b5': get_feature_depart_b1(),
        }[phi]

        backbone_cls = {
            'b0': mit_b0, 'b1': mit_b1, 'b2': mit_b2,
            'b3': mit_b3, 'b4': mit_b4, 'b5': mit_b5,
        }[phi]
        self.backbone_vi = backbone_cls(pretrained, num_experts=num_experts, depths=depths)
        self.backbone_ir = backbone_cls(pretrained, num_experts=num_experts, depths=depths)
        self.embedding_dim   = {
            'b0': 256, 'b1': 256, 'b2': 768,
            'b3': 768, 'b4': 768, 'b5': 768,
        }[phi]

        self.decode_head = FusionHead(out_channel=128, in_channels=self.in_channels, embedding_dim=self.embedding_dim)

        self.specFusion = MultiModalGraphFusion(layer_dims=self.in_channels)
        self.baseFusion = baseFusion(layer_dims=self.in_channels)

        self.f_det_all = fused_all_features_for_detect(layer_dims=self.in_channels)
        self.decoder = resnet50_Decoder(2048) #fused4是512的，分别是64, 128, 320, 512
        # -----------------------------------------------------------------#
        #   对获取到的特征进行上采样，进行分类预测和回归预测
        #   128, 128, 64 -> 128, 128, 64 -> 128, 128, num_classes
        #                -> 128, 128, 64 -> 128, 128, 2
        #                -> 128, 128, 64 -> 128, 128, 2
        # -----------------------------------------------------------------#
        self.head = resnet50_Head(channel=64, num_classes=num_classes)


        self.conv_out = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=3, kernel_size=3, stride=1, padding=1),
        )

        self._init_weights()

        self.fusion_feature = concat_fusion(layer_num=self.in_channels)

    def _init_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.head.cls_head[-1].weight.data.fill_(0)
        self.head.cls_head[-1].bias.data.fill_(-2.19)



    def forward(self, vi, ir, return_features = False, return_det_res=False):
        H, W = vi.size(2), vi.size(3)
        
        vi_feature = self.backbone_vi.forward(vi)
        ir_feature = self.backbone_ir.forward(ir)

        vi_base, vi_de = self.features_depart.forward(vi_feature)
        ir_base, ir_de = self.features_depart.forward(ir_feature)

        fused_base = self.baseFusion(vi_base, ir_base)
        fused_de = self.specFusion(vi_de, ir_de)

        # fused = add_fusion(fused_base, fused_de)
        fused = self.fusion_feature(fused_base, fused_de)


        x = self.decode_head.forward(fused)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        x = self.conv_out(x)

        det_res = self.head(self.decoder(self.f_det_all(fused[0], fused[1], fused[2], fused[3])))


        if return_features:
            return x, vi_base, ir_base, vi_de, ir_de
        elif return_det_res:
            return x, vi_base, ir_base, vi_de, ir_de, det_res
        else:
            return x

if __name__ == '__main__':
    model = Net(phi='b1').cuda()
    img = torch.randn(1,3,224,224).cuda()
    model(img, img)
