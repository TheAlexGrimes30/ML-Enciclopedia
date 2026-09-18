import random

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

class SyntheticGenerationDataset(Dataset):
    def __init__(
            self,
            n_samples: int = 600,
            image_size: int = 64,
            num_classes: int = 3,
            seed: int = 42
    ):
        super().__init__()

        if num_classes != 3:
            raise ValueError("This synthetic dataset is designed for 3 classes")

        self.n_samples = n_samples
        self.image_size = image_size

        generator = torch.Generator()
        generator.manual_seed(seed)

        images = []
        labels = []

        for _ in range(n_samples):
            label = int(
                torch.randint(
                    low=0,
                    high=num_classes,
                    size=(1,),
                    generator=generator
                ).item()
            )

            image = torch.randn(
                3,
                image_size,
                image_size,
                generator=generator
            ) * 0.05

            center_y = int(
                torch.randint(
                    low=image_size // 3,
                    high=(2 * image_size) // 3,
                    size=(1,),
                    generator=generator
                ).item()
            )

            center_x = int(
                torch.randint(
                    low=image_size // 3,
                    high=(2 * image_size) // 3,
                    size=(1,),
                    generator=generator
                ).item()
            )

            thickness = image_size // 10
            length = image_size // 2

            if label == 0:
                x1 = max(0, center_x - thickness // 2)
                x2 = min(image_size, center_x + thickness // 2)
                y1 = max(0, center_y - length // 2)
                y2 = min(image_size, center_y + length // 2)

                image[0, y1:y2, x1:x2] += 1.0

            elif label == 1:
                x1 = max(0, center_x - length // 2)
                x2 = min(image_size, center_x + length // 2)
                y1 = max(0, center_y - thickness // 2)
                y2 = min(image_size, center_y + thickness // 2)

                image[1, y1:y2, x1:x2] += 1.0

            else:
                side = image_size // 3

                x1 = max(0, center_x - side // 2)
                x2 = min(image_size, center_x + side // 2)
                y1 = max(0, center_y - side // 2)
                y2 = min(image_size, center_y + side // 2)

                image[2, y1:y2, x1:x2] += 1.0

            image = image.clamp(0.0, 1.0)

            images.append(image)
            labels.append(label)

        self.images = torch.stack(images)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

class ConvEncoder(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            base_channels: int = 32,
            latent_dim: int = 32,
            image_size: int = 64
    ):
        super().__init__()

        if image_size != 64:
            raise ValueError("This implementation expects image_size=64")

        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=base_channels,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                in_channels=base_channels,
                out_channels=base_channels * 2,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                in_channels=base_channels * 2,
                out_channels=base_channels * 4,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                in_channels=base_channels * 4,
                out_channels=base_channels * 8,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True)
        )

        self.feature_dim = base_channels * 8 * 4 * 4

        self.mu = nn.Linear(
            self.feature_dim,
            latent_dim
        )

        self.logvar = nn.Linear(
            self.feature_dim,
            latent_dim
        )

    def forward(self, x: torch.Tensor):
        x = self.features(x)
        x = x.flatten(start_dim=1)

        mu = self.mu(x)
        logvar = self.logvar(x)

        return mu, logvar

class ConvDecoder(nn.Module):
    def __init__(
            self,
            out_channels: int = 3,
            base_channels: int = 32,
            latent_dim: int = 32,
            image_size: int = 64
    ):
        super().__init__()

        if image_size != 64:
            raise ValueError("This implementation expects image_size=64")

        self.base_channels = base_channels
        self.projection = nn.Linear(
            latent_dim,
            base_channels * 8 * 4 * 4
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=base_channels * 8,
                out_channels=base_channels * 4,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                in_channels=base_channels * 4,
                out_channels=base_channels * 2,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                in_channels=base_channels * 2,
                out_channels=base_channels,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                in_channels=base_channels,
                out_channels=out_channels,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.Sigmoid()
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.projection(z)
        x = x.view(z.size(0), self.base_channels * 8, 4, 4)
        x = self.decoder(x)

        return x

class ConvolutionalVAE(nn.Module):
    def __init__(
            self,
            image_size: int = 64,
            in_channels: int = 3,
            latent_dim: int = 32,
            base_channels: int = 32
    ):
        super().__init__()

        self.encoder = ConvEncoder(
            in_channels=in_channels,
            base_channels=base_channels,
            latent_dim=latent_dim,
            image_size=image_size
        )

        self.decoder = ConvDecoder(
            out_channels=in_channels,
            base_channels=base_channels,
            latent_dim=latent_dim,
            image_size=image_size
        )

    def reparameterize(
            self,
            mu: torch.Tensor,
            logvar: torch.Tensor
    ) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)

        return mu + epsilon * std

    def encode(self, x: torch.Tensor):
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)

        return reconstruction, mu, logvar

    @torch.no_grad()
    def sample(
            self,
            num_samples: int,
            device
    ) -> torch.Tensor:
        z = torch.randn(num_samples, self.encoder.mu.out_features, device=device)

        return self.decode(z)

def vae_loss(
        reconstruction: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        beta: float = 1.0
):
    batch_size = x.size(0)

    reconstruction_loss = F.mse_loss(
        reconstruction,
        x,
        reduction="sum"
    ) / batch_size

    kl_loss = -0.5 * torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp()
    ) / batch_size

    total_loss = reconstruction_loss + beta * kl_loss

    return total_loss, reconstruction_loss, kl_loss


def train_epoch(
        model: nn.Module,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device,
        beta: float = 1.0
):
    model.train()

    total_loss = 0.0
    total_reconstruction = 0.0
    total_kl = 0.0
    total = 0

    for images, _ in tqdm(dataloader, desc="Training"):
        images = images.to(device)

        optimizer.zero_grad()

        reconstruction, mu, logvar = model(images)

        loss, reconstruction_loss, kl_loss = vae_loss(
            reconstruction=reconstruction,
            x=images,
            mu=mu,
            logvar=logvar,
            beta=beta
        )

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)

        total_loss += loss.item() * batch_size
        total_reconstruction += reconstruction_loss.item() * batch_size
        total_kl += kl_loss.item() * batch_size
        total += batch_size

    return (
        total_loss / total,
        total_reconstruction / total,
        total_kl / total
    )


@torch.no_grad()
def evaluate(
        model: nn.Module,
        dataloader: DataLoader,
        device,
        beta: float = 1.0
):
    model.eval()

    total_loss = 0.0
    total_reconstruction = 0.0
    total_kl = 0.0
    total = 0

    for images, _ in tqdm(dataloader, desc="Validation"):
        images = images.to(device)

        reconstruction, mu, logvar = model(images)

        loss, reconstruction_loss, kl_loss = vae_loss(
            reconstruction=reconstruction,
            x=images,
            mu=mu,
            logvar=logvar,
            beta=beta
        )

        batch_size = images.size(0)

        total_loss += loss.item() * batch_size
        total_reconstruction += reconstruction_loss.item() * batch_size
        total_kl += kl_loss.item() * batch_size
        total += batch_size

    return (
        total_loss / total,
        total_reconstruction / total,
        total_kl / total
    )


def train_model(
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        device,
        epochs: int = 10,
        lr: float = 1e-3,
        beta: float = 1.0
):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr
    )

    for epoch in range(epochs):
        train_total, train_reconstruction, train_kl = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            beta=beta
        )

        test_total, test_reconstruction, test_kl = evaluate(
            model=model,
            dataloader=test_loader,
            device=device,
            beta=beta
        )

        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Train total loss:          {train_total:.4f}")
        print(f"Train reconstruction loss: {train_reconstruction:.4f}")
        print(f"Train KL loss:             {train_kl:.4f}")
        print(f"Test total loss:           {test_total:.4f}")
        print(f"Test reconstruction loss:  {test_reconstruction:.4f}")
        print(f"Test KL loss:              {test_kl:.4f}")
        print()


@torch.no_grad()
def reconstruct_image(
        model: nn.Module,
        image: torch.Tensor,
        device
):
    model.eval()

    image = image.unsqueeze(0).to(device)

    reconstruction, mu, logvar = model(image)

    return (
        reconstruction.squeeze(0).cpu(),
        mu.squeeze(0).cpu(),
        logvar.squeeze(0).cpu()
    )


@torch.no_grad()
def generate_images(
        model: nn.Module,
        num_samples: int,
        device
) -> torch.Tensor:
    model.eval()

    return model.sample(
        num_samples=num_samples,
        device=device
    ).cpu()


@torch.no_grad()
def save_reconstruction_grid(
        model: nn.Module,
        dataset: Dataset,
        device,
        path: str,
        num_images: int = 8
):
    model.eval()

    originals = []
    reconstructions = []

    for index in range(num_images):
        image, _ = dataset[index]
        reconstruction, _, _ = reconstruct_image(
            model=model,
            image=image,
            device=device
        )

        originals.append(image)
        reconstructions.append(reconstruction)

    originals = torch.stack(originals)
    reconstructions = torch.stack(reconstructions)

    grid = torch.cat([originals, reconstructions], dim=0)
    save_image(grid, path, nrow=num_images)


@torch.no_grad()
def save_generated_grid(
        model: nn.Module,
        device,
        path: str,
        num_images: int = 16
):
    samples = generate_images(
        model=model,
        num_samples=num_images,
        device=device
    )

    save_image(samples, path, nrow=4)


def count_parameters(
        model: nn.Module
) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def main():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    train_dataset = SyntheticGenerationDataset(
        n_samples=600,
        image_size=64,
        num_classes=3,
        seed=42
    )

    test_dataset = SyntheticGenerationDataset(
        n_samples=300,
        image_size=64,
        num_classes=3,
        seed=1337
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False
    )

    image, true_label = test_dataset[0]

    print("\n=======================================")
    print("CUSTOM CONVOLUTIONAL VAE")
    print("=======================================")

    custom_vae = ConvolutionalVAE(
        image_size=64,
        in_channels=3,
        latent_dim=32,
        base_channels=32
    ).to(device)

    print(custom_vae)
    print("Trainable parameters:", count_parameters(custom_vae))

    train_model(
        model=custom_vae,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        epochs=10,
        lr=1e-3,
        beta=1.0
    )

    reconstruction, mu, logvar = reconstruct_image(
        model=custom_vae,
        image=image,
        device=device
    )

    generated_samples = generate_images(
        model=custom_vae,
        num_samples=8,
        device=device
    )

    print("\nCustom VAE reconstruction")
    print("Image shape:", image.shape)
    print("True class:", int(true_label))
    print("Latent mean shape:", mu.shape)
    print("Latent logvar shape:", logvar.shape)
    print("Reconstruction shape:", reconstruction.shape)
    print("Generated samples shape:", generated_samples.shape)

    save_reconstruction_grid(
        model=custom_vae,
        dataset=test_dataset,
        device=device,
        path="custom_vae_reconstructions.png",
        num_images=8
    )

    save_generated_grid(
        model=custom_vae,
        device=device,
        path="custom_vae_samples.png",
        num_images=16
    )

    custom_test_total, custom_test_reconstruction, custom_test_kl = evaluate(
        model=custom_vae,
        dataloader=test_loader,
        device=device,
        beta=1.0
    )

    print("Saved: custom_vae_reconstructions.png")
    print("Saved: custom_vae_samples.png")
    print(f"Custom VAE total loss:           {custom_test_total:.4f}")
    print(f"Custom VAE reconstruction loss:  {custom_test_reconstruction:.4f}")
    print(f"Custom VAE KL loss:              {custom_test_kl:.4f}")


if __name__ == "__main__":
    main()
