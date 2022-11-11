"""
ResNet 1D
The code is based on the original ResNet implementation from torchvision.models.resnet
"""

import torch.nn as nn


def conv3(in_planes, out_planes, kernel_size, stride=1, dilation=1):
    return nn.Conv1d(
        in_planes,
        out_planes,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
        bias=False,
    )


class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(
        self, in_planes, out_planes, kernel_size, stride=1, dilation=1, downsample=None
    ):
        super(BasicBlock1D, self).__init__()
        self.conv1 = conv3(in_planes, out_planes, kernel_size, stride, dilation)
        self.bn1 = nn.BatchNorm1d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3(out_planes, out_planes, kernel_size)
        self.bn2 = nn.BatchNorm1d(out_planes)
        self.stride = stride
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck1D(nn.Module):
    expansion = 4

    def __init__(
        self, in_planes, out_planes, kernel_size, stride=1, dilation=1, downsample=None
    ):
        super(Bottleneck1D, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, out_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_planes)
        self.conv2 = conv3(out_planes, out_planes, kernel_size, stride, dilation)
        self.bn2 = nn.BatchNorm1d(out_planes)
        self.conv3 = nn.Conv1d(
            out_planes, out_planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm1d(out_planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet1D(nn.Module):
    def __init__(
        self,
        n_features,
        n_targets,
        block_type=BasicBlock1D,
        group_sizes=[2, 2, 2, 2],
        base_plane=64,
        zero_init_residual=False,
        window_size=150,
        **kwargs
    ):
        super(ResNet1D, self).__init__()
        self.base_plane = base_plane
        self.inplanes = self.base_plane

        self.activation = nn.ReLU()

        self.n_targets = n_targets
        self.window_size = window_size

        # Input module
        self.input_block = nn.Sequential(
            nn.Conv1d(
                n_features,
                self.inplanes,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            ),
            nn.BatchNorm1d(self.inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        # Residual groups
        self.planes = [self.base_plane * (2 ** i) for i in range(len(group_sizes))]
        kernel_size = kwargs.get("kernel_size", 3)
        strides = [1] + [2] * (len(group_sizes) - 1)
        dilations = [1] * len(group_sizes)
        groups = [
            self._make_residual_group1d(
                block_type,
                self.planes[i],
                kernel_size,
                group_sizes[i],
                strides[i],
                dilations[i],
            )
            for i in range(len(group_sizes))
        ]
        self.residual_groups = nn.Sequential(*groups)

        # Output
        self.output_block = nn.Sequential(
            nn.Linear(512, self.window_size, bias=True),
            self.activation,
            nn.Linear(self.window_size, self.window_size, bias=True),
            self.activation,
            nn.Linear(self.window_size, self.n_targets, bias=True),
        )

        self._initialize(zero_init_residual)

    def _make_residual_group1d(
        self, block_type, planes, kernel_size, blocks, stride=1, dilation=1
    ):
        downsample = None
        if stride != 1 or self.inplanes != planes * block_type.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(
                    self.inplanes,
                    planes * block_type.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm1d(planes * block_type.expansion),
            )
        layers = []
        layers.append(
            block_type(
                self.inplanes,
                planes,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                downsample=downsample,
            )
        )
        self.inplanes = planes * block_type.expansion
        for _ in range(1, blocks):
            layers.append(block_type(self.inplanes, planes, kernel_size=kernel_size))

        return nn.Sequential(*layers)

    def _initialize(self, zero_init_residual):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck1D):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock1D):
                    nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x):
        x = self.input_block(x)
        x = self.residual_groups(x)

        # Reshape to fit FC layers
        x = x.mean(-1)
        x = x.view(x.shape[0], -1)
        x = self.output_block(x)

        return x

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def main():
    import torch
    from torchviz import make_dot, make_dot_from_trace

    data = torch.rand(48, 1, 140)

    window_size = data.shape[-1]
    n_features = data.shape[1]

    network = ResNet1D(
        n_features=n_features,
        n_targets=6,
        window_size=window_size,
        group_sizes=[1, 1, 1, 1],
        base_plane=64,
        kernel_size=3,
    )

    output = network(data)

    print(network.get_num_params())

    # make_dot(output.mean(), params=dict(network.named_parameters())).render("resnet_1d", format="png")


if __name__ == "__main__":
    main()
