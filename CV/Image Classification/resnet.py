import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18
from tqdm import tqdm

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

class ResidualBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int = 1
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(
            out_channels
        )

        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            bias=False
        )

        self.bn2 = nn.BatchNorm2d(
            out_channels
        )

        if (stride != 1 or in_channels != out_channels):
            self.shortcut = nn.Sequential(

                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),

                nn.BatchNorm2d(
                    out_channels
                )
            )

        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, num_classes: int = 3):
        super().__init__()

        self.stem = nn.Sequential(

            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(32),

            nn.ReLU()
        )

        self.layer1 = nn.Sequential(

            ResidualBlock(
                in_channels=32,
                out_channels=32
            ),

            ResidualBlock(
                in_channels=32,
                out_channels=32
            )
        )

        self.layer2 = nn.Sequential(

            ResidualBlock(
                in_channels=32,
                out_channels=64,
                stride=2
            ),

            ResidualBlock(
                in_channels=64,
                out_channels=64
            )
        )

        self.layer3 = nn.Sequential(

            ResidualBlock(
                in_channels=64,
                out_channels=128,
                stride=2
            ),

            ResidualBlock(
                in_channels=128,
                out_channels=128
            )
        )

        self.avgpool = nn.AdaptiveAvgPool2d(
            output_size=(1, 1)
        )

        self.fc = nn.Linear(
            in_features=128,
            out_features=num_classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
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
        self.image_size = image_size

        generator = torch.Generator()

        generator.manual_seed(
            seed
        )

        images = []
        labels = []

        stripe_width = (
                image_size // 8
        )

        centers = [
            image_size // 4,
            image_size // 2,
            3 * image_size // 4
        ]

        for i in range(
                n_samples
        ):

            label = (
                    i % 3
            )

            image = torch.randn(
                3,
                image_size,
                image_size,
                generator=generator
            ) * 0.08

            center = centers[
                label
            ]

            left = (
                    center
                    - stripe_width // 2
            )

            right = (
                    center
                    + stripe_width // 2
            )

            image[
                :,
                :,
                left:right
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
        model: nn.Module,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
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

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        predictions = logits.argmax(
            dim=1
        )

        correct += (
                predictions == labels
        ).sum().item()

        total += labels.size(0)

    return (
        total_loss / total,
        correct / total
    )

@torch.no_grad()
def evaluate(
        model: nn.Module,
        dataloader: DataLoader,
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
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        device,
        epochs: int = 5,
        lr: float = 1e-3
):

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr
    )

    for epoch in range(epochs):
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
        model: nn.Module,
        image: torch.Tensor,
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
        "CUSTOM SIMPLE RESNET"
    )

    print(
        "======================================="
    )

    simple_resnet = ResNet(
        num_classes=3
    ).to(
        device
    )

    print(
        simple_resnet
    )

    train_model(
        model=simple_resnet,
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
        model=simple_resnet,
        image=image,
        device=device
    )

    print(
        "\nSimpleResNet inference"
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
        "TORCHVISION RESNET18"
    )

    print(
        "======================================="
    )

    torchvision_resnet = resnet18(
        weights=None
    )

    torchvision_resnet.fc = nn.Linear(
        in_features=512,
        out_features=3
    )

    torchvision_resnet = torchvision_resnet.to(
        device
    )

    print(
        torchvision_resnet
    )

    train_model(
        model=torchvision_resnet,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        epochs=2,
        lr=1e-4
    )

    prediction, probabilities = inference(
        model=torchvision_resnet,
        image=image,
        device=device
    )

    print(
        "\nTorchvision ResNet18 inference"
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