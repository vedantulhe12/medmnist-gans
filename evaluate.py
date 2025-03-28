import torch
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.fid import FrechetInceptionDistance
from medmnist import ChestMNIST
import torchvision.transforms as transforms

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load real images
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
real_dataset = ChestMNIST(root="./data", split="test", download=True, transform=transform)
real_loader = torch.utils.data.DataLoader(real_dataset, batch_size=64, shuffle=False)

# Compute Metrics
def compute_metrics():
    inception = InceptionScore().to(device)
    fid = FrechetInceptionDistance().to(device)

    # Get real images
    real_images = next(iter(real_loader))[0].to(device)
    real_images = (real_images * 255).byte()  # Convert to uint8
    real_images = real_images.repeat(1, 3, 1, 1)  # Convert grayscale to RGB if needed
    fid.update(real_images, real=True)

    for gan in ["LS-GAN", "WGAN", "WGAN-GP"]:
        generator = Generator().to(device)
        generator.load_state_dict(torch.load(f"models/{gan}_generator.pth"))
        generator.eval()

        # Generate fake images
        fake_images = torch.cat([generator(torch.randn(64, 100).to(device)) for _ in range(10)], dim=0)
        fake_images = (fake_images + 1) / 2  # Rescale to [0,1]
        fake_images = (fake_images * 255).byte()
        fake_images = fake_images.repeat(1, 3, 1, 1)

        inception.update(fake_images)
        fid.update(fake_images, real=False)

        score, _ = inception.compute()
        fid_value = fid.compute()

        print(f"{gan} - Inception Score: {score.item()}, FID: {fid_value.item()}")

compute_metrics()
