import random

import numpy as np
import torch
from torch import nn
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

class ConvGenerator(nn.Module):
    def __init__(
            self,
            latent_dim: int = 100,
            out_channels: int = 3,
            base_channels: int = 64,
            image_size: int = 64
    ):
        super().__init__()

        if image_size != 64:
            raise ValueError("This implementation expects image_size=64")

        self.latent_dim = latent_dim
        self.base_channels = base_channels

        self.projection = nn.Sequential(
            nn.Linear(latent_dim, base_channels * 8 * 4 * 4),
            nn.ReLU(inplace=True)
        )

        self.model = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=base_channels * 8,
                out_channels=base_channels * 4,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                in_channels=base_channels * 4,
                out_channels=base_channels * 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                in_channels=base_channels * 2,
                out_channels=base_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                in_channels=base_channels,
                out_channels=out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.Sigmoid()
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.projection(z)
        x = x.view(z.size(0), self.base_channels * 8, 4, 4)

        return self.model(x)

class ConvDiscriminator(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            base_channels: int = 64,
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
                padding=1,
                bias=False
            ),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(
                in_channels=base_channels,
                out_channels=base_channels * 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(
                in_channels=base_channels * 2,
                out_channels=base_channels * 4,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(
                in_channels=base_channels * 4,
                out_channels=base_channels * 8,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.classifier = nn.Linear(
            base_channels * 8 * 4 * 4,
            1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.flatten(start_dim=1)
        x = self.classifier(x)

        return x

class ConvolutionalGAN(nn.Module):
    def __init__(
            self,
            image_size: int = 64,
            in_channels: int = 3,
            latent_dim: int = 100,
            base_channels: int = 64
    ):
        super().__init__()

        self.latent_dim = latent_dim

        self.generator = ConvGenerator(
            latent_dim=latent_dim,
            out_channels=in_channels,
            base_channels=base_channels,
            image_size=image_size
        )

        self.discriminator = ConvDiscriminator(
            in_channels=in_channels,
            base_channels=base_channels,
            image_size=image_size
        )

    def sample_noise(self, batch_size: int, device) -> torch.Tensor:
        return torch.randn(batch_size, self.latent_dim, device=device)

    def generate(self, batch_size: int, device) -> torch.Tensor:
        noise = self.sample_noise(batch_size=batch_size, device=device)
        return self.generator(noise)

    def discriminate(self, images: torch.Tensor) -> torch.Tensor:
        return self.discriminator(images)

class GANLoss:
    def __init__(self):
        self.criterion = nn.BCEWithLogitsLoss()

    def discriminator_loss(
            self,
            real_logits: torch.Tensor,
            fake_logits: torch.Tensor
    ):
        real_targets = torch.ones_like(real_logits)
        fake_targets = torch.zeros_like(fake_logits)

        real_loss = self.criterion(real_logits, real_targets)
        fake_loss = self.criterion(fake_logits, fake_targets)
        total_loss = real_loss + fake_loss

        return total_loss, real_loss, fake_loss

    def generator_loss(
            self,
            fake_logits: torch.Tensor
    ):
        targets = torch.ones_like(fake_logits)
        return self.criterion(fake_logits, targets)


@torch.no_grad()
def discriminator_accuracy(
        real_logits: torch.Tensor,
        fake_logits: torch.Tensor
):
    real_predictions = (torch.sigmoid(real_logits) >= 0.5).float()
    fake_predictions = (torch.sigmoid(fake_logits) < 0.5).float()

    real_accuracy = real_predictions.mean().item()
    fake_accuracy = fake_predictions.mean().item()

    return real_accuracy, fake_accuracy


def train_epoch(
        model: nn.Module,
        dataloader: DataLoader,
        generator_optimizer: torch.optim.Optimizer,
        discriminator_optimizer: torch.optim.Optimizer,
        loss_fn: GANLoss,
        device
):
    model.train()

    total_discriminator_loss = 0.0
    total_generator_loss = 0.0
    total_real_loss = 0.0
    total_fake_loss = 0.0
    total_real_accuracy = 0.0
    total_fake_accuracy = 0.0
    total = 0

    for real_images, _ in tqdm(dataloader, desc="Training"):
        real_images = real_images.to(device)
        batch_size = real_images.size(0)

        discriminator_optimizer.zero_grad()

        real_logits = model.discriminate(real_images)

        noise = model.sample_noise(batch_size=batch_size, device=device)
        fake_images = model.generator(noise).detach()
        fake_logits = model.discriminate(fake_images)

        discriminator_loss, real_loss, fake_loss = loss_fn.discriminator_loss(
            real_logits=real_logits,
            fake_logits=fake_logits
        )

        discriminator_loss.backward()
        discriminator_optimizer.step()

        generator_optimizer.zero_grad()

        noise = model.sample_noise(batch_size=batch_size, device=device)
        generated_images = model.generator(noise)
        generated_logits = model.discriminate(generated_images)

        generator_loss = loss_fn.generator_loss(
            fake_logits=generated_logits
        )

        generator_loss.backward()
        generator_optimizer.step()

        real_accuracy, fake_accuracy = discriminator_accuracy(
            real_logits=real_logits.detach(),
            fake_logits=fake_logits.detach()
        )

        total_discriminator_loss += discriminator_loss.item() * batch_size
        total_generator_loss += generator_loss.item() * batch_size
        total_real_loss += real_loss.item() * batch_size
        total_fake_loss += fake_loss.item() * batch_size
        total_real_accuracy += real_accuracy * batch_size
        total_fake_accuracy += fake_accuracy * batch_size
        total += batch_size

    return {
        "discriminator_loss": total_discriminator_loss / total,
        "generator_loss": total_generator_loss / total,
        "real_loss": total_real_loss / total,
        "fake_loss": total_fake_loss / total,
        "real_accuracy": total_real_accuracy / total,
        "fake_accuracy": total_fake_accuracy / total,
    }


@torch.no_grad()
def evaluate(
        model: nn.Module,
        dataloader: DataLoader,
        loss_fn: GANLoss,
        device
):
    model.eval()

    total_discriminator_loss = 0.0
    total_generator_loss = 0.0
    total_real_loss = 0.0
    total_fake_loss = 0.0
    total_real_accuracy = 0.0
    total_fake_accuracy = 0.0
    total = 0

    for real_images, _ in tqdm(dataloader, desc="Validation"):
        real_images = real_images.to(device)
        batch_size = real_images.size(0)

        real_logits = model.discriminate(real_images)

        noise = model.sample_noise(batch_size=batch_size, device=device)
        fake_images = model.generator(noise)
        fake_logits = model.discriminate(fake_images)

        discriminator_loss, real_loss, fake_loss = loss_fn.discriminator_loss(
            real_logits=real_logits,
            fake_logits=fake_logits
        )

        generator_loss = loss_fn.generator_loss(
            fake_logits=fake_logits
        )

        real_accuracy, fake_accuracy = discriminator_accuracy(
            real_logits=real_logits,
            fake_logits=fake_logits
        )

        total_discriminator_loss += discriminator_loss.item() * batch_size
        total_generator_loss += generator_loss.item() * batch_size
        total_real_loss += real_loss.item() * batch_size
        total_fake_loss += fake_loss.item() * batch_size
        total_real_accuracy += real_accuracy * batch_size
        total_fake_accuracy += fake_accuracy * batch_size
        total += batch_size

    return {
        "discriminator_loss": total_discriminator_loss / total,
        "generator_loss": total_generator_loss / total,
        "real_loss": total_real_loss / total,
        "fake_loss": total_fake_loss / total,
        "real_accuracy": total_real_accuracy / total,
        "fake_accuracy": total_fake_accuracy / total,
    }


def train_model(
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        device,
        epochs: int = 10,
        lr: float = 2e-4,
        betas=(0.5, 0.999)
):
    loss_fn = GANLoss()

    generator_optimizer = torch.optim.Adam(
        model.generator.parameters(),
        lr=lr,
        betas=betas
    )

    discriminator_optimizer = torch.optim.Adam(
        model.discriminator.parameters(),
        lr=lr,
        betas=betas
    )

    for epoch in range(epochs):
        train_metrics = train_epoch(
            model=model,
            dataloader=train_loader,
            generator_optimizer=generator_optimizer,
            discriminator_optimizer=discriminator_optimizer,
            loss_fn=loss_fn,
            device=device
        )

        test_metrics = evaluate(
            model=model,
            dataloader=test_loader,
            loss_fn=loss_fn,
            device=device
        )

        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Train discriminator loss: {train_metrics['discriminator_loss']:.4f}")
        print(f"Train generator loss:     {train_metrics['generator_loss']:.4f}")
        print(f"Train real accuracy:      {train_metrics['real_accuracy']:.4f}")
        print(f"Train fake accuracy:      {train_metrics['fake_accuracy']:.4f}")
        print(f"Test discriminator loss:  {test_metrics['discriminator_loss']:.4f}")
        print(f"Test generator loss:      {test_metrics['generator_loss']:.4f}")
        print(f"Test real accuracy:       {test_metrics['real_accuracy']:.4f}")
        print(f"Test fake accuracy:       {test_metrics['fake_accuracy']:.4f}")
        print()


@torch.no_grad()
def generate_images(
        model: nn.Module,
        num_samples: int,
        device
) -> torch.Tensor:
    model.eval()
    return model.generate(batch_size=num_samples, device=device).cpu()


@torch.no_grad()
def discriminate_image(
        model: nn.Module,
        image: torch.Tensor,
        device
):
    model.eval()

    image = image.unsqueeze(0).to(device)
    logit = model.discriminate(image)
    probability = torch.sigmoid(logit)

    return float(probability.item())


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


@torch.no_grad()
def save_real_grid(
        dataset: Dataset,
        path: str,
        num_images: int = 16
):
    images = [dataset[index][0] for index in range(num_images)]
    images = torch.stack(images)
    save_image(images, path, nrow=4)


def count_parameters(model: nn.Module) -> int:
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

    save_real_grid(
        dataset=test_dataset,
        path="real_samples.png",
        num_images=16
    )

    print("\n=======================================")
    print("CUSTOM CONVOLUTIONAL GAN")
    print("=======================================")

    custom_gan = ConvolutionalGAN(
        image_size=64,
        in_channels=3,
        latent_dim=100,
        base_channels=64
    ).to(device)

    print(custom_gan)
    print("Trainable parameters:", count_parameters(custom_gan))

    train_model(
        model=custom_gan,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        epochs=10,
        lr=2e-4,
        betas=(0.5, 0.999)
    )

    generated_samples = generate_images(
        model=custom_gan,
        num_samples=8,
        device=device
    )

    real_probability = discriminate_image(
        model=custom_gan,
        image=image,
        device=device
    )

    print("\nCustom GAN generation")
    print("Example real image shape:", image.shape)
    print("True class:", int(true_label))
    print("Generated samples shape:", generated_samples.shape)
    print("Discriminator probability for one real image:", real_probability)

    save_generated_grid(
        model=custom_gan,
        device=device,
        path="custom_gan_samples.png",
        num_images=16
    )

    custom_metrics = evaluate(
        model=custom_gan,
        dataloader=test_loader,
        loss_fn=GANLoss(),
        device=device
    )

    print("Saved: real_samples.png")
    print("Saved: custom_gan_samples.png")


    print(f"Custom GAN discriminator loss:  {custom_metrics['discriminator_loss']:.4f}")
    print(f"Custom GAN generator loss:      {custom_metrics['generator_loss']:.4f}")
    print(f"Custom GAN real accuracy:       {custom_metrics['real_accuracy']:.4f}")
    print(f"Custom GAN fake accuracy:       {custom_metrics['fake_accuracy']:.4f}")

if __name__ == "__main__":
    main()
