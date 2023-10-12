# test the saved generator model

# load the model
import torch
import torch.cuda as cuda
from torch import nn

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

model = Generator(128, 784)
model.load_state_dict(torch.load('./model/gen.pth'))

device = torch.device('cpu' if not cuda.is_available() else 'cuda:0')

model.eval()
model.to(device)

# generate a random image
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 10, figsize=(10, 1))

for i in range(10):
    noise = torch.randn((1, 128)).to(device)
    generated_img = model(noise).cpu().detach().numpy().reshape(28, 28)

    ax[i].imshow(generated_img, cmap='gray')
    ax[i].axis('off')

plt.show()