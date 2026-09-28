import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from . import wavelet


class SpatialGatedFusionBlock(nn.Module):
    def __init__(self, channels):
        super(SpatialGatedFusionBlock, self).__init__()
        reduction = max(4, channels // 8)
        self.channels = channels
        self.attention = nn.Sequential(
            nn.Conv2d(channels * 2, channels // reduction, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels // reduction),
            nn.ReLU(inplace=True),
            # --- Change Start ---
            # 输出 2*C 通道
            nn.Conv2d(channels // reduction, channels * 2, 1, bias=False),
            # --- Change End ---
            nn.Sigmoid()
        )
        self.final_conv = nn.Conv2d(channels, channels, 1)

    def forward(self, v, i):
        combined = torch.cat([v, i], dim=1)
        gates = self.attention(combined)  # (B, 2*C, H, W)
        # --- Change Start ---
        gate_v, gate_i = torch.split(gates, self.channels, dim=1)  # (B, C, H, W) each
        # print(gate_v, gate_i)
        gated_fusion = gate_v * v + gate_i * i
        gated_fusion = self.final_conv(gated_fusion)  # 可选
        # --- Change End ---
        return gated_fusion


class WTConv2d_VIF(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bias=True, wt_levels=1, wt_type='db1'):
        super(WTConv2d_VIF, self).__init__()

        # assert in_channels == out_channels

        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride
        self.dilation = 1

        self.wt_filter, self.iwt_filter = wavelet.create_2d_wavelet_filter(wt_type, in_channels, in_channels,
                                                                           torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)

        self.base_conv = nn.Conv2d(in_channels, in_channels, kernel_size, padding='same', stride=1, dilation=1,
                                   groups=in_channels, bias=bias)
        self.base_scale = _ScaleModule([1, in_channels, 1, 1])

        self.wavelet_convs = nn.ModuleList(
            [nn.Conv2d(in_channels * 4, in_channels * 4, kernel_size, padding='same', stride=1, dilation=1,
                       groups=in_channels * 4, bias=False) for _ in range(self.wt_levels)]
        )
        self.wavelet_scale = nn.ModuleList(
            [_ScaleModule([1, in_channels * 4, 1, 1], init_scale=0.1) for _ in range(self.wt_levels)]
        )

        if self.stride > 1:
            self.do_stride = nn.AvgPool2d(kernel_size=1, stride=stride)
        else:
            self.do_stride = None

        self.fusion = SpatialGatedFusionBlock(self.in_channels)

        self.final_conv = nn.Conv2d(self.in_channels*2, out_channels, 1)

    def forward_once(self, x, y):

        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []

        y_ll_in_levels = []
        y_h_in_levels = []

        curr_x_ll = x
        curr_y_ll = y

        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)

            curr_x = wavelet.wavelet_2d_transform(curr_x_ll, self.wt_filter)
            curr_y = wavelet.wavelet_2d_transform(curr_y_ll, self.iwt_filter)
            curr_x_ll = curr_x[:, :, 0, :, :]
            curr_y_ll = curr_y[:, :, 0, :, :]

            shape_x = curr_x.shape
            shape_y = curr_y.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
            curr_y_tag = curr_y.reshape(shape_y[0], shape_y[1] * 4, shape_y[3], shape_y[4])
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
            curr_y_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_y_tag))
            curr_x_tag = curr_x_tag.reshape(shape_x)
            curr_y_tag = curr_y_tag.reshape(shape_y)

            x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :])
            x_h_in_levels.append(curr_x_tag[:, :, 1:4, :, :])
            y_ll_in_levels.append(curr_y_tag[:, :, 0, :, :])
            y_h_in_levels.append(curr_y_tag[:, :, 1:4, :, :])

        next_x_ll = 0

        for i in range(self.wt_levels - 1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_y_ll = y_ll_in_levels.pop()
            curr_shape = shapes_in_levels.pop()
            # print(curr_x_ll.shape)
            curr_x_ll = self.fusion(curr_x_ll, curr_y_ll)

            curr_x_ll = curr_x_ll + next_x_ll

            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
            next_x_ll = wavelet.inverse_2d_wavelet_transform(curr_x, self.iwt_filter)

            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]

        x_tag = next_x_ll
        assert len(x_ll_in_levels) == 0

        x = self.base_scale(self.base_conv(x))
        x = x + x_tag

        if self.do_stride is not None:
            x = self.do_stride(x)

        return x

    def forward(self, x, y):
        f1 = self.forward_once(x, y)
        f2 = self.forward_once(y, x)
        return self.final_conv(torch.cat([f1, f2], dim=1))


class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None

    def forward(self, x):
        return torch.mul(self.weight, x)






if __name__ == '__main__':
    model = WTConv2d_VIF(in_channels=16, out_channels=16)
    x = torch.randn(1, 16, 640, 640)
    x = model(x, x)
    print(x.shape)
