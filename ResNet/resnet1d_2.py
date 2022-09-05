import torch
import torch.nn as nn

def conv(in_channels, out_channels, kernel_size, stride):
    return nn.Conv1d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        padding_mode=kernel_size//2,
        stride=stride,
        bias=False
        )


class Block(nn.Module):
    def __init__(self, 
                in_channels: int, 
                out_channels: int =32,
                activation: str ="relu",  # ReLU and LeakyReLU
                kernel_size: int =7,
                stride: int =1,) -> None:
        super(Block, self).__init__()
        self.expansion = 1
        
        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "relu":
            self.activation = nn.LeakyReLU(inplace=True)

        self.conv1 = nn.Sequential(
                    conv(in_channels, out_channels, kernel_size, stride),
                    nn.BatchNorm1d(out_channels),
                    self.activation
                )

        self.conv2 = nn.Sequential(
                    conv(in_channels, 2*out_channels, kernel_size, stride),
                    nn.BatchNorm1d(2*out_channels),
                )
    
    def forward(self, x):
        # for skip connection
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)

        out += residual

        return self.activation(out)


class ResNet1D(nn.Module):
    def __init__(self):
        super(ResNet1D, self).__init__()

        pass

