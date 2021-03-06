"""Network architectures for MuZero

Author: Masahiro Hayashi

This script defines the network architectures for MuZero, which consists of
3 components: Representation, Dynamic, and Prediction

The file is organized into 2 sections:
    - building blocks for defining the network
    - components of MuZero
    - main architecture of MuZero
"""
import torch
from torch import nn

###############################################################################
# Building Blocks
###############################################################################
class ConvBnReLU(nn.Module):
    """Convoutional-Batch Normalization-ReLU block
    """
    def __init__(
        self, in_dim, out_dim, filter_size=3, stride=1, padding=1, noReLU=False
    ):
        self.name = 'Conv-BN-ReLU'
        super(ConvBnReLU, self).__init__()
        self.noReLU = noReLU
        self.conv = nn.Conv2d(in_dim, out_dim, filter_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_dim)
        if not noReLU:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if not self.noReLU:
            x = self.relu(x)
        return x

class ConvBnLeakyReLU(ConvBnReLU):
    """Convoutional-Batch Normalization-LeakyReLU block
    """
    def __init__(self, in_dim, out_dim, filter_size=3, stride=1, padding=1):
        super(ConvBnLeakyReLU, self).__init__(in_dim, out_dim)
        self.name = 'Conv-BN-LeakyReLU'
        self.relu = nn.LeakyReLU(2e-1, inplace=True)

class ResidualBlock(nn.Module):
    """Basic Residiual Block
    """
    def __init__(
        self, in_dim, out_dim, padding=1, padding_type='zero',
        downsample=None, stride=1, use_dropout=False
    ):
        self.name = 'Basic Residual Block'
        super(ResidualBlock, self).__init__()
        self.downsample = downsample
        # padding options
        _pad = {
            'reflect': nn.ReflectionPad2d(padding),
            'replicate': nn.ReflectionPad2d(padding)
        }
        self.block = []
        if padding_type == 'zero':
            self.padding = 1
        else:
            self.padding = 0
            self.pad = _pad.get(padding_type, None)
            if self.pad is None:
                NotImplementedError(
                    f'padding [{padding_type}] is not implemented'
                )
        # conv-bn-relu 1
        if self.padding == 0:
            self.block.append(self.pad)
        conv1 = ConvBnReLU(in_dim, out_dim, 3, stride, padding)
        self.block.append(conv1)
        if use_dropout:
            dropout = nn.Dropout2d(0.5) if use_dropout else None
            self.block.append(dropout)
        # conv-bn-relu 2
        if self.padding == 0:
            self.block.append(self.pad)
        conv2 = ConvBnReLU(out_dim, out_dim, 3, stride, padding, noReLU=True)
        self.block.append(conv2)
        self.block = nn.Sequential(*self.block)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        identity = x
        # if not self.downsample is None:
        #     identity = self.downsample(identity)
        residual = self.block(x)
        out = identity + residual
        out = self.relu(out)
        return out

class ResidualBlocks(nn.Module):
    """Multiple residual blocks
    """
    def __init__(
        self, n_blocks, in_dim, out_dim, padding=1, padding_type='zero',
        downsample=None, stride=1, use_dropout=False
    ):
        self.name = "Residual Blocks"
        super(ResidualBlocks, self).__init__()
        self.blocks = [
            ResidualBlock(
                in_dim, out_dim, padding, padding_type,
                downsample, stride, use_dropout
            ) for _ in range(n_blocks)
        ]
        self.blocks = nn.Sequential(*self.blocks)

    def forward(self, x):
        return self.blocks(x)

class MLP(nn.Module):
    """Multi-layer Perceptron
    """
    def __init__(self, in_dim, out_dim, h_dim=128, n_layers=2):
        self.name = "Multi-layer Perceptron"
        super(MLP, self).__init__()
        self.layers = []
        self.layers.append(nn.Linear(in_dim, h_dim))
        for _ in range(n_layers-2):
            self.layers.append(nn.Linear(h_dim, h_dim))
        self.layers.append(nn.Linear(h_dim, out_dim))
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.layers(x)

###############################################################################
# MuZero Networks
###############################################################################
class RepresentationNet(nn.Module):
    """Representation Network
    """
    def __init__(self):
        self.name = 'Representation Network'
        super(RepresentationNet, self).__init__()
        self.conv1 = ConvBnReLU(128, 128, stride=2)
        self.resblocks1 = ResidualBlocks(2, 128, 128)
        self.conv2 = ConvBnReLU(128, 256, stride=2)
        self.resblocks2 = ResidualBlocks(3, 256, 256)
        self.avg_pool1 = nn.AvgPool2d(3, stride=2, padding=1)
        self.resblocks3 = ResidualBlocks(3, 256, 256)
        self.avg_pool2 = nn.AvgPool2d(3, stride=2, padding=1)


    def forward(self, x):
        x = self.conv1(x)
        x = self.resblocks1(x)
        x = self.conv2(x)
        x = self.resblocks2(x)
        x = self.avg_pool1(x)
        x = self.resblocks3(x)
        x = self.avg_pool2(x)
        return x

class DynamicNet(nn.Module):
    """Dynamic Network
    """
    def __init__(self):
        self.name = 'Dynamic Network'
        super(DynamicNet, self).__init__()
        self.conv1 = ConvBnReLU(128, 128, stride=2)
        self.resblocks1 = ResidualBlocks(2, 128, 128)
        self.conv2 = ConvBnReLU(128, 256, stride=2)
        self.resblocks2 = ResidualBlocks(3, 256, 256)
        self.avg_pool1 = nn.AvgPool2d(3, stride=2, padding=1)
        self.resblocks3 = ResidualBlocks(3, 256, 256)
        self.avg_pool2 = nn.AvgPool2d(3, stride=2, padding=1)
        self.conv3 = ConvBnReLU(256, 256, stride=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.resblocks1(x)
        x = self.conv2(x)
        x = self.resblocks2(x)
        x = self.avg_pool1(x)
        x = self.resblocks3(x)
        x = self.avg_pool2(x)
        return x

###############################################################################
# For testing
###############################################################################

if __name__ == '__main__':
    layers = [0, 4, 8, 12, 16]
    X = torch.rand((1, 128, 96, 96))
    # rep_net = RepresentationNet()
    # print(rep_net)
    dyna_net = DynamicNet()
    print(dyna_net)
    y = dyna_net(X)
    print(y.size())
    # mlp = MLP(10, 10)
    # print(mlp)

