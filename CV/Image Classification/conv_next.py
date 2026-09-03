import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import convnext_tiny
from tqdm import tqdm

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

class ConvNeXtBlock(nn.Module):
    def __init__(
            self,
            channels: int,
            expansion: int = 4
    ):
        super().__init__()

        hidden_channels = (channels * expansion)

        self.depthwise = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=7,
            groups=channels,
            bias=True
        )

        self.norm = nn.LayerNorm(channels)

        self.expand = nn.Linear(
            in_features=channels,
            out_features=hidden_channels
        )

        self.activation = nn.GELU()

        self.project = nn.Linear(
            in_features=hidden_channels,
            out_features=channels
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        x = self.depthwise(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.expand(x)
        x = self.activation(x)
        x = self.project(x)
        x = x.permute(0, 3, 1, 2)
        x = x + identity
        return x

class ConvNeXt(nn.Module):
    def __init__(self, num_classes: int = 3):
        super().__init__()

        self.stem = nn.Sequential(

            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=4,
                stride=4
            )
        )

        self.downsample1 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=2,
            stride=2
        )

        self.stage1 = nn.Sequential(

            ConvNeXtBlock(
                channels=32
            ),

            ConvNeXtBlock(
                channels=32
            )
        )

        self.stage2 = nn.Sequential(

            ConvNeXtBlock(
                channels=64
            ),

            ConvNeXtBlock(
                channels=64
            )
        )

        self.downsample2 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=2,
            stride=2
        )

        self.stage3 = nn.Sequential(

            ConvNeXtBlock(
                channels=128
            ),

            ConvNeXtBlock(
                channels=128
            )
        )

        self.avgpool = nn.AdaptiveAvgPool2d(
            output_size=(1, 1)
        )

        self.norm = nn.LayerNorm(128)

        self.classifier = nn.Linear(
            in_features=128,
            out_features=num_classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.downsample1(x)
        x = self.stage2(x)
        x = self.downsample2(x)
        x = self.stage3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.norm(x)
        x = self.classifier(x)
        return x

class SyntheticDataset(Dataset):

    def __init__(
            self,
            n_samples: int = 600,
            image_size: int = 64,
            seed: int = 42
    ):
        super().__init__()

        self.n_samples = n_samples

        generator = torch.Generator()

        generator.manual_seed(
            seed
        )

        images = []
        labels = []

        stripe_width = (
                image_size // 8
        )

        center = (
                image_size // 2
        )

        left = (
                center
                - stripe_width // 2
        )

        right = (
                center
                + stripe_width // 2
        )

        for i in range(
                n_samples
        ):

            label = i % 3

            image = torch.randn(
                3,
                image_size,
                image_size,
                generator=generator
            ) * 0.08

            # Vertical stripe

            if label == 0:

                image[
                    :,
                    :,
                    left:right
                ] += 1.0

            # Horizontal stripe

            elif label == 1:

                image[
                    :,
                    left:right,
                    :
                ] += 1.0

            # Cross

            else:

                image[
                    :,
                    :,
                    left:right
                ] += 1.0

                image[
                    :,
                    left:right,
                    :
                ] += 1.0

            image = image.clamp(
                0,
                1
            )

            images.append(
                image
            )

            labels.append(
                label
            )

        self.images = torch.stack(
            images
        )

        self.labels = torch.tensor(
            labels,
            dtype=torch.long
        )

    def __len__(
            self
    ):
        return self.n_samples

    def __getitem__(
            self,
            index
    ):

        return (
            self.images[index],
            self.labels[index]
        )

def train_epoch(
        model,
        dataloader,
        optimizer,
        criterion,
        device
):

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(
            dataloader,
            desc="Training"
    ):

        images = images.to(
            device
        )

        labels = labels.to(
            device
        )

        optimizer.zero_grad()

        logits = model(
            images
        )

        loss = criterion(
            logits,
            labels
        )

        loss.backward()

        optimizer.step()

        total_loss += (
                loss.item()
                * images.size(0)
        )

        predictions = logits.argmax(
            dim=1
        )

        correct += (
                predictions
                == labels
        ).sum().item()

        total += labels.size(0)

    return (
        total_loss / total,
        correct / total
    )

@torch.no_grad()
def evaluate(
        model,
        dataloader,
        criterion,
        device
):

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(
            dataloader,
            desc="Validation"
    ):

        images = images.to(
            device
        )

        labels = labels.to(
            device
        )

        logits = model(
            images
        )

        loss = criterion(
            logits,
            labels
        )

        predictions = logits.argmax(
            dim=1
        )

        total_loss += (
                loss.item()
                * images.size(0)
        )

        correct += (
                predictions
                == labels
        ).sum().item()

        total += labels.size(0)

    return (
        total_loss / total,
        correct / total
    )

def train_model(
        model,
        train_loader,
        test_loader,
        device,
        epochs: int = 5,
        lr: float = 1e-3
):

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4
    )

    for epoch in range(
            epochs
    ):

        train_loss, train_acc = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device
        )

        test_loss, test_acc = evaluate(
            model=model,
            dataloader=test_loader,
            criterion=criterion,
            device=device
        )

        print(
            f"Epoch {epoch + 1}/{epochs}"
        )

        print(
            f"Train loss: {train_loss:.4f}"
        )

        print(
            f"Train accuracy: {train_acc:.4f}"
        )

        print(
            f"Test loss: {test_loss:.4f}"
        )

        print(
            f"Test accuracy: {test_acc:.4f}"
        )

        print()

@torch.no_grad()
def inference(
        model,
        image,
        device
):

    model.eval()

    image = image.unsqueeze(
        dim=0
    )

    image = image.to(
        device
    )

    logits = model(
        image
    )

    probabilities = torch.softmax(
        logits,
        dim=1
    )

    prediction = probabilities.argmax(
        dim=1
    )

    return (
        prediction.item(),
        probabilities.cpu()
    )

def main():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device
    )

    train_dataset = SyntheticDataset(
        n_samples=600,
        image_size=64,
        seed=42
    )

    test_dataset = SyntheticDataset(
        n_samples=300,
        image_size=64,
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

    print(
        "\n"
        "======================================="
    )

    print(
        "CUSTOM SIMPLE CONVNEXT"
    )

    print(
        "======================================="
    )

    custom_model = ConvNeXt(
        num_classes=3
    ).to(
        device
    )

    print(
        custom_model
    )

    train_model(
        model=custom_model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        epochs=5,
        lr=1e-3
    )

    image, true_label = (
        test_dataset[0]
    )

    prediction, probabilities = inference(
        model=custom_model,
        image=image,
        device=device
    )

    print(
        "\nCustom ConvNeXt inference"
    )

    print(
        "True label:",
        true_label.item()
    )

    print(
        "Prediction:",
        prediction
    )

    print(
        "Probabilities:",
        probabilities
    )

    print(
        "\n"
        "======================================="
    )

    print(
        "TORCHVISION CONVNEXT TINY"
    )

    print(
        "======================================="
    )

    torchvision_model = convnext_tiny(
        weights=None
    )

    torchvision_model.classifier[2] = nn.Linear(
        in_features=768,
        out_features=3
    )

    torchvision_model = torchvision_model.to(
        device
    )

    print(
        torchvision_model
    )

    train_model(
        model=torchvision_model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        epochs=2,
        lr=1e-3
    )

    prediction, probabilities = inference(
        model=torchvision_model,
        image=image,
        device=device
    )

    print(
        "\nTorchvision ConvNeXt Tiny inference"
    )

    print(
        "True label:",
        true_label.item()
    )

    print(
        "Prediction:",
        prediction
    )

    print(
        "Probabilities:",
        probabilities
    )


if __name__ == "__main__":
    main()