import torch

from model import Generator, Discriminator


def check_generator():
    x = torch.randn((1, 3, 256, 256))
    gen = Generator(in_channels=3)
    print(gen(x).shape)


def check_discriminator():
    x = torch.randn((1, 3, 256, 256))
    disc = Discriminator(in_channels=3)
    print(disc(x).shape)


def check_both():
    gen = Generator(in_channels=3, ini_features=64, num_residual_blocks=16)
    disc = Discriminator(in_channels=3, features=[
                         64, 64, 128, 128, 256, 256, 512, 512])
    low_res = 96
    x = torch.randn((10, 3, low_res, low_res)) # expect 96 x 4
    gen_out = gen(x)
    disc_out = disc(gen_out)

    print(gen_out.shape)
    print(disc_out.shape)


if __name__ == "__main__":
    # check_generator()
    # check_discriminator()
    check_both()
