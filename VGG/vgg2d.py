import torch
import torch.nn as nn


class Block(nn.Module):
    """
        A block containing convolutinal, relu and maxpool layers
        according to vgg paper.
    """
    def __init__(self, in_channels, out_channels, num_layers):
        super().__init__()

        self.relu = nn.ReLU()

        # First layer of the block
        layers = [nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    ),
                    self.relu
                ]

        # Add other layers if required
        for _ in range(num_layers):
            layers.append(nn.Conv2d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    )
                )
            layers.append(self.relu)

        # Do maxpooling
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        # Now let's add these layers sequtially
        self.block = nn.Sequential(*layers)
    

    def forward(self, x):
        return self.block(x)


class VGG2D(nn.Module):
    def __init__(self, in_channels: int, output_size: int, VGG_type: list[:int]):
        super().__init__()
        self.feature_maps = [in_channels, 64, 128, 256, 512, 512]

        self.convs = nn.ModuleList([])

        # Create all blocks together
        for i in range(len(VGG_type)):
            self.convs.append(Block(
                in_channels=self.feature_maps[i],
                out_channels=self.feature_maps[i+1],
                num_layers=VGG_type[i]
            ))

        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

        self. fc1 = nn.Sequential(
                        nn.Linear(7*7*512, 4096),
                        self.dropout, 
                        self.relu
                    )
        
        self. fc2 = nn.Sequential(
                        nn.Linear(4096, 4096),
                        self.dropout, 
                        self.relu
                    )
        
        self. fc3 = nn.Linear(4096, output_size)
         
        self.init_weights()
    
    
    def forward(self, x):
        for block in self.convs:
            x = block(x)
        
        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x
    
    def init_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.01)
                nn.init.constant_(layer.bias, 0)


def get_vgg(in_channels, output_size, architecture="vgg19"):
    VGGs = dict(
            vgg11 = [1, 1, 2, 2, 2],
            vgg13 = [2, 2, 2, 2, 2],
            vgg16 = [2, 2, 3, 3, 3],
            vgg19 = [2, 2, 4, 4, 4],
        )
    
    net = VGG2D(in_channels=in_channels, 
            output_size=output_size, 
            VGG_type=VGGs[architecture]
            )
    
    return net
    
    
# Let's see if this works
def main():
    import os
    from torchinfo import summary
    try:
        os.system("clear")
    except:
        pass
    
    data = torch.rand(8, 3, 224, 224)

    vgg = get_vgg(in_channels=data.shape[1], output_size=6, architecture="vgg11")

    summary(vgg, input_size=(2, 3, 224, 224))

    return


if __name__ == "__main__":
    main()
