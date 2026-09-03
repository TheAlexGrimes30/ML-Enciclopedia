import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import efficientnet_b0
from tqdm import tqdm

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

class SqueezeExcitation(nn.Module):
    def __init__(
            self,
            channels: int,
            reduction: int = 4
    ):
        super().__init__()

        hidden_channels = max(1, channels // reduction)

        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.fc = nn.Sequential(
            nn.Conv2d(
                in_channels=channels,
                out_channels=hidden_channels,
                kernel_size=1
            ),

            nn.SiLU(),

            nn.Conv2d(
                in_channels=hidden_channels,
                out_channels=channels,
                kernel_size=1
            ),

            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.pool(x)
        weights = self.fc(weights)
        return x * weights

class MBConv(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            expand_ratio: int = 4,
            stride: int =1
    ):
        super().__init__()

        hidden_channels = (in_channels * expand_ratio)

        self.use_residual = (
                stride == 1
                and in_channels == out_channels
        )

        self.expand = nn.Sequential(

            nn.Conv2d(
                in_channels=in_channels,
                out_channels=hidden_channels,
                kernel_size=1,
                bias=False
            ),

            nn.BatchNorm2d(
                hidden_channels
            ),

            nn.SiLU()
        )

        self.depthwise = nn.Sequential(

            nn.Conv2d(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=hidden_channels,
                bias=False
            ),

            nn.BatchNorm2d(
                hidden_channels
            ),

            nn.SiLU()
        )

        self.se = SqueezeExcitation(
            channels=hidden_channels,
            reduction=4
        )

        self.project = nn.Sequential(

            nn.Conv2d(
                in_channels=hidden_channels,
                out_channels=out_channels,
                kernel_size=1,
                bias=False
            ),

            nn.BatchNorm2d(
                out_channels
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        x = self.expand(x)
        x = self.depthwise(x)
        x = self.se(x)
        x = self.project(x)

        if self.use_residual:
            x = x + identity

        return x

class EfficientNet(nn.Module):
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
            nn.SiLU()
        )

        self.features = nn.Sequential(
            MBConv(
                in_channels=32,
                out_channels=32,
                expand_ratio=2,
                stride=1
            ),

            MBConv(
                in_channels=32,
                out_channels=64,
                expand_ratio=4,
                stride=2
            ),

            MBConv(
                in_channels=64,
                out_channels=64,
                expand_ratio=4,
                stride=1
            ),

            MBConv(
                in_channels=64,
                out_channels=128,
                expand_ratio=4,
                stride=2
            ),

            MBConv(
                in_channels=128,
                out_channels=128,
                expand_ratio=4,
                stride=1
            )
        )

        self.head = nn.Sequential(

            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=1,
                bias=False
            ),

            nn.BatchNorm2d(
                256
            ),

            nn.SiLU()
        )

        self.avgpool = nn.AdaptiveAvgPool2d(
            output_size=(1, 1)
        )

        self.classifier = nn.Sequential(

            nn.Dropout(
                p=0.2
            ),

            nn.Linear(
                in_features=256,
                out_features=num_classes
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.features(x)
        x = self.head(x)
        x = self.avgpool(x)

        x = torch.flatten(x, start_dim=1)

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

            if label == 0:

                image[
                    :,
                    :,
                    left:right
                ] += 1.0

            elif label == 1:

                image[
                    :,
                    left:right,
                    :
                ] += 1.0

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
                predictions == labels
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
                predictions == labels
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
        epochs=5,
        lr=1e-3
):

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr
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
        "CUSTOM SIMPLE EFFICIENTNET"
    )

    print(
        "======================================="
    )

    simple_efficientnet = EfficientNet(
        num_classes=3
    ).to(
        device
    )

    print(
        simple_efficientnet
    )

    train_model(
        model=simple_efficientnet,
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
        model=simple_efficientnet,
        image=image,
        device=device
    )

    print(
        "\nSimple EfficientNet inference"
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
        "TORCHVISION EFFICIENTNET-B0"
    )

    print(
        "======================================="
    )

    torchvision_model = efficientnet_b0(

        weights=None
    )

    torchvision_model.classifier[1] = nn.Linear(
        in_features=1280,
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
        "\nTorchvision EfficientNet-B0 inference"
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

