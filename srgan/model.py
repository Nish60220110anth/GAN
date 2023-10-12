import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, isdescriminator=False, use_batchnorm=True, use_activation=True, **kwargs):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              bias=not use_batchnorm, **kwargs)
        self.batchnorm = nn.BatchNorm2d(
            out_channels) if use_batchnorm else nn.Identity()
        self.activation = nn.LeakyReLU(0.2, inplace=True) if isdescriminator else nn.PReLU(
            num_parameters=out_channels) if use_activation else nn.Identity()

    def forward(self, x):
        return self.activation(self.batchnorm(self.conv(x)))


class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        # contains two blocks but without pixel wise addition for skip connection
        super(ResidualBlock, self).__init__()
        self.block1 = ConvBlock(in_channels, in_channels, isdescriminator=False, use_batchnorm=True, use_activation=True,
                                kernel_size=3, padding=1, stride=1)
        self.block2 = ConvBlock(in_channels, in_channels, isdescriminator=False, use_batchnorm=True, use_activation=False,
                                kernel_size=3, padding=1, stride=1)

    def forward(self, x):
        return x + self.block2(self.block1(x))  # skip connection x + F(x)


class UpSampleBlock(nn.Module):
    def __init__(self, in_channels, scale_factor, **kwargs):
        super(UpSampleBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels, in_channels *
                              scale_factor**2, kernel_size=3, padding=1, stride=1)  # scale_factor**2 = H*scale_factor*W*scale_factor
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        # after pixel shuffle, the no of channels remains the same
        self.prelu = nn.PReLU(num_parameters=in_channels)
        # as they are distributed btw height and width

    def forward(self, x):
        return self.prelu(self.pixel_shuffle(self.conv(x)))


class Generator(nn.Module):
    def __init__(self, in_channels, ini_features=64, num_residual_blocks=16):
        super(Generator, self).__init__()

        self.initial = ConvBlock(in_channels, ini_features, isdescriminator=False,
                                 use_batchnorm=False, use_activation=True, kernel_size=9, padding=4, stride=1)  # 4 = (9-1)/2
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(ini_features) for _ in range(num_residual_blocks)])
        self.mid = ConvBlock(ini_features, ini_features,
                             isdescriminator=False, use_batchnorm=True, use_activation=False, kernel_size=3, padding=1, stride=1)
        self.upsample_blocks = nn.Sequential(
            *[UpSampleBlock(ini_features, scale_factor=2) for _ in range(2)])
        self.final = nn.Conv2d(ini_features, in_channels,
                               kernel_size=9, padding=4, stride=1)  # image dimension increased by 4

    def forward(self, x):
        initial = self.initial(x)
        residual = self.residual_blocks(initial)
        mid_out = self.mid(residual) + initial  # skip connection from initial
        out = self.upsample_blocks(mid_out)
        return torch.tanh(self.final(out))


class Discriminator(nn.Module):
    def __init__(self, in_channels, features=[64, 64, 128, 128, 256, 256, 512, 512]):
        super(Discriminator, self).__init__()

        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, features[0],
                      kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        layers = []
        for i in range(len(features)-1):
            layers.append(ConvBlock(features[i], features[i+1], isdescriminator=True,
                                    use_batchnorm=True, use_activation=True, kernel_size=3, padding=1, stride=(i+1) % 2+1))

        self.residuals = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(),
            nn.Linear(512*6*6, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1)
        )

    def forward(self, x):
        return self.classifier(self.residuals(self.initial(x)))
