import torch
import torch.nn as nn
import torchvision

from model import Generator, Discriminator


def train():
    pass


def main():
    # load dataset and crop to get low res images

    dataloader = torchvision.datasets.ImageFolder(root="data", transform=torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(64),
        torchvision.transforms.ToTensor()
    ]))


if __name__ == "__main__":
    main()
