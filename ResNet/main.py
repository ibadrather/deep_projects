import torch
import torch.nn as nn

class Resnet1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode):
        super(Resnet1d, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.conv2(out)
        out += residual
        out = self.bn(out)
        out = self.relu(out)
        return out


class Resnet1dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode):
        super(Resnet1dBlock, self).__init__()
        self.resnet1d = Resnet1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        self.conv = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        residual = x
        out = self.resnet1d(x)
        out = self.conv(out)
        out += residual
        out = self.bn(out)
        out = self.relu(out)
        return out