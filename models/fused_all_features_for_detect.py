import torch
import torch.nn.functional as F
import torch.nn as nn

class Downsample(nn.Module):
    '''
    分辨率下采样2倍，通道数加倍
    '''
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class fused_all_features_for_detect(nn.Module):
    def __init__(self,layer_dims=[64, 128, 320, 512]):
        super(fused_all_features_for_detect, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=layer_dims[0]*8+layer_dims[1]*4+layer_dims[2]*2+layer_dims[3], out_channels=2048, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(in_channels=2048*2, out_channels=2048, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels=2048, out_channels=2048, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(2048)
        self.d11 = Downsample(layer_dims[0])
        self.d12 = Downsample(layer_dims[0]*2)
        self.d13 = Downsample(layer_dims[0]*4)

        self.d21 = Downsample(layer_dims[1])
        self.d22 = Downsample(layer_dims[1]*2)

        self.d31 = Downsample(layer_dims[2])


    def forward(self, x1, x2, x3, x4):
        '''
        64, 128, 320, 512 原始的输入通道数
        # 考虑只使用320和512的深层特征，x3, x4
        :param x:
        :return:
        '''
        x1 = self.d13(self.d12(self.d11(x1)))
        x2 = self.d22(self.d21(x2))
        x3 = self.d31(x3)

        all = torch.cat((x1, x2,x3, x4), dim=1)
        y = self.conv1(all)
        y = self.bn(y)
        y = self.conv3(y)
        return y

