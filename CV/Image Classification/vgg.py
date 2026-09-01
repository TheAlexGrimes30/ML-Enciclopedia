import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import vgg16
from tqdm import tqdm

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

class VGG(nn.Module):

    def __init__(
            self,
            num_classes: int = 3
    ):
        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),

            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),

            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),

            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),

            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),
        )

        self.avgpool = nn.AdaptiveAvgPool2d(
            output_size=(4, 4)
        )

        self.classifier = nn.Sequential(

            nn.Linear(
                128 * 4 * 4,
                64
            ),

            nn.ReLU(),

            nn.Dropout(
                p=0.5
            ),

            nn.Linear(
                64,
                num_classes
            )
        )

    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:

        x = self.features(x)

        x = self.avgpool(x)

        x = torch.flatten(
            x,
            start_dim=1
        )

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
        self.image_size = image_size

        generator = torch.Generator()
        generator.manual_seed(seed)

        images = []
        labels = []

        stripe_width = image_size // 8

        centers = [
            image_size // 4,
            image_size // 2,
            3 * image_size // 4
        ]

        for i in range(n_samples):

            label = i % 3

            image = torch.randn(
                3,
                image_size,
                image_size,
                generator=generator
            ) * 0.08

            center = centers[label]

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

            images.append(image)
            labels.append(label)

        self.images = torch.stack(
            images
        )

        self.labels = torch.tensor(
            labels,
            dtype=torch.long
        )

    def __len__(self):
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

    for images, labels in tqdm(dataloader, desc="Training"):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)
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
        model: nn.Module,
        dataloader: DataLoader,
        criterion,
        device
):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(dataloader, desc="Validation"):
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

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
        "CUSTOM SIMPLE VGG"
    )

    print(
        "======================================="
    )

    simple_vgg = VGG(
        num_classes=3
    ).to(device)

    print(
        simple_vgg
    )

    train_model(
        model=simple_vgg,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        epochs=5,
        lr=1e-3
    )

    image, true_label = test_dataset[0]

    prediction, probabilities = inference(
        model=simple_vgg,
        image=image,
        device=device
    )

    print(
        "SimpleVGG inference"
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
        "TORCHVISION VGG16"
    )

    print(
        "======================================="
    )


    torchvision_vgg = vgg16(
        weights=None
    )

    torchvision_vgg.classifier[6] = nn.Linear(
        in_features=4096,
        out_features=3
    )

    torchvision_vgg = torchvision_vgg.to(
        device
    )

    print(
        torchvision_vgg
    )

    train_model(
        model=torchvision_vgg,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        epochs=2,
        lr=1e-4
    )

    prediction, probabilities = inference(
        model=torchvision_vgg,
        image=image,
        device=device
    )

    print(
        "\nTorchvision VGG16 inference"
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