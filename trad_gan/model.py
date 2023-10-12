import torch
from torch import nn
from torch import cuda
from torch import optim
from torch.utils.data import DataLoader

import torchvision as tvision
from torchvision import datasets as datasets
from torchvision import transforms as transforms
from torch.utils.tensorboard import writer

device = torch.device('cpu' if not cuda.is_available() else 'cuda:0')

batch_size = 64
num_epochs = 30
lr = 0.01
z_dim = 128
image_dim = 28 * 28 * 1  # 784 MNIST
fixed_noise = torch.randn((batch_size, z_dim)).to(device)


class NeuralBlock(nn.Module):
    def __init__(self, in_dim, out_dim, with_dropout, dropout, **kwargs):
        super(NeuralBlock, self).__init__()

        self.network = nn.Linear(
            in_features=in_dim, out_features=out_dim, bias=True)
        self.with_dropout = with_dropout
        if with_dropout:
            self.dropout = nn.Dropout(p=dropout)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x = self.network(x)

        if self.with_dropout:
            x = self.dropout(x)

        x = self.lrelu(x)

        return x


class Descriminator(nn.Module):
    def __init__(self, in_dim):
        super(Descriminator, self).__init__()
        self.model = nn.Sequential(
            NeuralBlock(in_dim=in_dim, out_dim=1024,
                        with_dropout=True, dropout=0.3),
            NeuralBlock(in_dim=1024, out_dim=512,
                        with_dropout=True, dropout=0.3),
            NeuralBlock(in_dim=512, out_dim=300,
                        with_dropout=True, dropout=0.3),
            NeuralBlock(in_dim=300, out_dim=128,
                        with_dropout=True, dropout=0.3),
            NeuralBlock(in_dim=128, out_dim=64,
                        with_dropout=False, dropout=0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), 784)
        return self.model(x)


class Generator(nn.Module):
    def __init__(self, z_dim, out_dim):
        super(Generator, self).__init__()

        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(512, 720),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(720, 1024),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(1024, out_dim),
            nn.Tanh()  # [-1, 1]
        )

    def forward(self, x):
        return self.gen(x)


def init_weigths(module):
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight)


def main():
    disc = Descriminator(image_dim).to(device)
    gen = Generator(z_dim, image_dim).to(device)

    disc.apply(init_weigths)
    gen.apply(init_weigths)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.3,))
    ])

    dataset = datasets.MNIST(
        root='dataset/', transform=transform, download=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    opt_disc = optim.Adam(disc.parameters(), lr=lr, weight_decay=1e-6)
    opt_gen = optim.Adam(gen.parameters(), lr=lr, weight_decay=1e-6)

    criterion = nn.BCELoss().to(device=device)

    writer_fake = writer.SummaryWriter(f'logs/fake')
    writer_real = writer.SummaryWriter(f'logs/real')

    step = 0
    for epoch in range(num_epochs):
        for _, (real, _) in enumerate(loader):
            # loss: y*log(D(x)) + (1-y)*log(1-D(G(z))
            real = real.view(-1, 784).to(device)

            # Train discriminator
            latentz = torch.randn(batch_size, z_dim).to(device)
            fakes = gen(latentz)

            disc_real = disc(real).view(-1)
            loss_disc_real = criterion(
                disc_real, torch.ones_like(disc_real))  # log(D(x))

            disc_fake = disc(fakes).view(-1)
            loss_disc_fake = criterion(
                disc_fake, torch.zeros_like(disc_fake))  # log(1-D(G(z)))

            loss_disc = (loss_disc_real + loss_disc_fake) / 2

            disc.zero_grad()
            loss_disc.backward(retain_graph=True)
            opt_disc.step()

            # Train generator
            output = disc(fakes).view(-1)
            loss_gen = criterion(output, torch.ones_like(output))

            gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()

            # Tensorboard
            writer_fake.add_scalar('Loss', loss_gen.item(), global_step=step)
            writer_real.add_scalar('Loss', loss_disc.item(), global_step=step)

            step += 1

        print(
            f"Epoch [{epoch}/{num_epochs}] Loss D: {loss_disc:.4f}, Loss G: {loss_gen:.4f}")

        with torch.no_grad():
            fake = gen(fixed_noise).reshape(-1, 1, 28, 28)
            data = real.reshape(-1, 1, 28, 28)

            img_grid_fake = tvision.utils.make_grid(fake, normalize=True)
            img_grid_real = tvision.utils.make_grid(data, normalize=True)

            writer_fake.add_image(
                "Mnist Fake", img_grid_fake, global_step=step)
            writer_real.add_image(
                "Mnist Real", img_grid_real, global_step=step)

    writer_fake.close()
    writer_real.close()

    torch.save(gen.state_dict(), './drive/MyDrive/logs/gen.pth')
    torch.save(disc.state_dict(), './drive/MyDrive/logs/disc.pth')


if __name__ == "__main__":
    main()