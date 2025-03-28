import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.utils as vutils
from medmnist import ChestMNIST
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load MedMNIST Dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = ChestMNIST(root="./data", split="train", download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Generator Model
class Generator(nn.Module):
    def __init__(self, z_dim=100):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 28*28),
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z).view(-1, 1, 28, 28)

# Discriminator Model
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.net(x.view(x.size(0), -1))

# WGAN-GP Gradient Penalty
def gradient_penalty(D, real_data, fake_data):
    alpha = torch.rand(real_data.size(0), 1, 1, 1).to(device)
    interpolates = (alpha * real_data + (1 - alpha) * fake_data).requires_grad_(True)
    d_interpolates = D(interpolates)
    grad_outputs = torch.ones_like(d_interpolates)
    gradients = torch.autograd.grad(
        outputs=d_interpolates, inputs=interpolates, grad_outputs=grad_outputs,
        create_graph=True, retain_graph=True)[0]
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()

# Training Function
def train_gan(gan_type, num_epochs=50):
    writer = SummaryWriter(f"runs/{gan_type}")
    
    z_dim = 100
    generator = Generator(z_dim).to(device)
    discriminator = Discriminator().to(device)

    optim_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optim_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    for epoch in range(num_epochs):
        for real, _ in tqdm(train_loader):
            real = real.to(device)

            # Generate fake images
            z = torch.randn(real.size(0), z_dim).to(device)
            fake = generator(z)

            # Discriminator update
            optim_D.zero_grad()
            real_loss, fake_loss = 0, 0

            if gan_type == "LS-GAN":
                real_loss = 0.5 * ((discriminator(real) - 1) ** 2).mean()
                fake_loss = 0.5 * (discriminator(fake) ** 2).mean()
            elif gan_type == "WGAN":
                real_loss = -discriminator(real).mean()
                fake_loss = discriminator(fake).mean()
            elif gan_type == "WGAN-GP":
                real_loss = -discriminator(real).mean()
                fake_loss = discriminator(fake).mean()
                gp = gradient_penalty(discriminator, real, fake)
                loss_D = real_loss + fake_loss + 10 * gp
            else:
                raise ValueError("Invalid GAN type")

            loss_D = real_loss + fake_loss
            loss_D.backward()
            optim_D.step()

            # Generator update
            if epoch % 5 == 0:
                optim_G.zero_grad()
                fake = generator(z)
                loss_G = -discriminator(fake).mean() if gan_type in ["WGAN", "WGAN-GP"] else ((discriminator(fake) - 1) ** 2).mean()
                loss_G.backward()
                optim_G.step()

                # TensorBoard Logging
                writer.add_scalar("Loss/Discriminator", loss_D.item(), epoch)
                writer.add_scalar("Loss/Generator", loss_G.item(), epoch)

        # Save generated images
        vutils.save_image(fake[:25], f"generated/{gan_type}_epoch_{epoch}.png", normalize=True)

    torch.save(generator.state_dict(), f"models/{gan_type}_generator.pth")
    writer.close()

# Ensure directories exist
os.makedirs("generated", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Train all three GANs
for gan in ["LS-GAN", "WGAN", "WGAN-GP"]:
    train_gan(gan)
