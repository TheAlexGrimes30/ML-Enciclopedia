import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import vit_b_16
from tqdm import tqdm

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

class PatchEmbedding(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            image_size: int = 64,
            patch_size: int = 16,
            embed_dim: int = 128
    ):
        super().__init__()

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

        self.projection = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projection(x)

        x = x.flatten(
            start_dim=2
        )

        x = x.transpose(
            1,
            2
        )

        return x

class MultiHeadSelfAttention(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            dropout: float = 0.0
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(
            embed_dim,
            embed_dim * 3
        )

        self.attention_dropout = nn.Dropout(
            dropout
        )

        self.projection = nn.Linear(
            embed_dim,
            embed_dim
        )

        self.projection_dropout = nn.Dropout(
            dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens, embed_dim = x.shape

        qkv = self.qkv(x)

        qkv = qkv.reshape(
            batch_size,
            num_tokens,
            3,
            self.num_heads,
            self.head_dim
        )

        qkv = qkv.permute(
            2,
            0,
            3,
            1,
            4
        )

        query, key, value = qkv.unbind(
            dim=0
        )

        attention_scores = (
                    query @ key.transpose(-2, -1)
                    ) * self.scale

        attention = torch.softmax(
            attention_scores,
            dim=-1
        )

        attention = self.attention_dropout(
            attention
        )

        x = attention @ value

        x = x.transpose(
            1,
            2
        ).reshape(
            batch_size,
            num_tokens,
            embed_dim
        )

        x = self.projection(x)
        x = self.projection_dropout(x)

        return x

class MLP(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            mlp_dim: int,
            dropout: float = 0.0
    ):
        super().__init__()

        self.block = nn.Sequential(
            nn.Linear(
                embed_dim,
                mlp_dim
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                mlp_dim,
                embed_dim
            ),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class TransformerEncoderBlock(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            mlp_dim: int,
            dropout: float = 0.0
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(
            embed_dim
        )

        self.attention = MultiHeadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        self.norm2 = nn.LayerNorm(
            embed_dim
        )

        self.mlp = MLP(
            embed_dim=embed_dim,
            mlp_dim=mlp_dim,
            dropout=dropout
        )

    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:
        x = x + self.attention(
            self.norm1(x)
        )

        x = x + self.mlp(
            self.norm2(x)
        )

        return x

class VisionTransformer(nn.Module):
    def __init__(
            self,
            image_size: int = 64,
            patch_size: int = 16,
            in_channels: int = 3,
            num_classes: int = 3,
            embed_dim: int = 128,
            depth: int = 4,
            num_heads: int = 4,
            mlp_dim: int = 256,
            dropout: float = 0.1
    ):
        super().__init__()

        self.patch_embedding = PatchEmbedding(
            in_channels=in_channels,
            image_size=image_size,
            patch_size=patch_size,
            embed_dim=embed_dim
        )

        num_patches = self.patch_embedding.num_patches

        self.class_token = nn.Parameter(
            torch.zeros(
                1,
                1,
                embed_dim
            )
        )

        self.position_embedding = nn.Parameter(
            torch.zeros(
                1,
                num_patches + 1,
                embed_dim
            )
        )

        self.embedding_dropout = nn.Dropout(dropout)

        self.encoder = nn.Sequential(
            *[
                TransformerEncoderBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    dropout=dropout
                )
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(
            embed_dim
        )

        self.classifier = nn.Linear(
            embed_dim,
            num_classes
        )

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.trunc_normal_(
            self.position_embedding,
            std=0.02
        )

        nn.init.trunc_normal_(
            self.class_token,
            std=0.02
        )

        nn.init.normal_(
            self.classifier.weight,
            std=0.02
        )

        nn.init.zeros_(
            self.classifier.bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embedding(x)

        batch_size = x.size(0)

        class_token = self.class_token.expand(
            batch_size,
            -1,
            -1
        )

        x = torch.cat(
            [class_token, x],
            dim=1
        )

        x = x + self.position_embedding
        x = self.embedding_dropout(x)

        x = self.encoder(x)
        x = self.norm(x)

        class_representation = x[:, 0]

        logits = self.classifier(
            class_representation
        )

        return logits

class SyntheticClassificationDataset(Dataset):
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
                x1 = max(
                    0,
                    center_x - thickness // 2
                )
                x2 = min(
                    image_size,
                    center_x + thickness // 2
                )
                y1 = max(
                    0,
                    center_y - length // 2
                )
                y2 = min(
                    image_size,
                    center_y + length // 2
                )

                image[
                    0,
                    y1:y2,
                    x1:x2
                ] += 1.0

            elif label == 1:
                x1 = max(
                    0,
                    center_x - length // 2
                )
                x2 = min(
                    image_size,
                    center_x + length // 2
                )
                y1 = max(
                    0,
                    center_y - thickness // 2
                )
                y2 = min(
                    image_size,
                    center_y + thickness // 2
                )

                image[
                    1,
                    y1:y2,
                    x1:x2
                ] += 1.0

            else:
                side = image_size // 3

                x1 = max(
                    0,
                    center_x - side // 2
                )
                x2 = min(
                    image_size,
                    center_x + side // 2
                )
                y1 = max(
                    0,
                    center_y - side // 2
                )
                y2 = min(
                    image_size,
                    center_y + side // 2
                )

                image[
                    2,
                    y1:y2,
                    x1:x2
                ] += 1.0

            image = image.clamp(
                0.0,
                1.0
            )

            images.append(image)
            labels.append(label)

        self.images = torch.stack(images)
        self.labels = torch.tensor(
            labels,
            dtype=torch.long
        )

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        return (
            self.images[index],
            self.labels[index]
        )

def accuracy_score(
        logits: torch.Tensor,
        labels: torch.Tensor
) -> float:
    predictions = torch.argmax(
        logits,
        dim=1
    )

    accuracy = (
            predictions == labels
    ).float().mean()

    return accuracy.item()


def train_epoch(
        model: nn.Module,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion,
        device
):
    model.train()

    total_loss = 0.0
    total_accuracy = 0.0
    total = 0

    for images, labels in tqdm(
            dataloader,
            desc="Training"
    ):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(images)

        loss = criterion(
            logits,
            labels
        )

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)

        total_loss += (
                loss.item()
                * batch_size
        )

        total_accuracy += (
                accuracy_score(
                    logits,
                    labels
                )
                * batch_size
        )

        total += batch_size

    return (
        total_loss / total,
        total_accuracy / total
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
    total_accuracy = 0.0
    total = 0

    for images, labels in tqdm(
            dataloader,
            desc="Validation"
    ):
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)

        loss = criterion(
            logits,
            labels
        )

        batch_size = images.size(0)

        total_loss += (
                loss.item()
                * batch_size
        )

        total_accuracy += (
                accuracy_score(
                    logits,
                    labels
                )
                * batch_size
        )

        total += batch_size

    return (
        total_loss / total,
        total_accuracy / total
    )


def train_model(
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        device,
        epochs: int = 5,
        lr: float = 3e-4
):
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4
    )

    for epoch in range(epochs):
        train_loss, train_accuracy = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device
        )

        test_loss, test_accuracy = evaluate(
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
            f"Train accuracy: {train_accuracy:.4f}"
        )
        print(
            f"Test loss: {test_loss:.4f}"
        )
        print(
            f"Test accuracy: {test_accuracy:.4f}"
        )
        print()


@torch.no_grad()
def inference(
        model: nn.Module,
        image: torch.Tensor,
        device
):
    model.eval()

    image = image.unsqueeze(0).to(device)

    logits = model(image)

    probabilities = torch.softmax(
        logits,
        dim=1
    )

    predicted_class = torch.argmax(
        probabilities,
        dim=1
    )

    return (
        probabilities.squeeze(0).cpu(),
        int(predicted_class.item())
    )


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

    print(
        "Device:",
        device
    )

    train_dataset = SyntheticClassificationDataset(
        n_samples=600,
        image_size=64,
        num_classes=3,
        seed=42
    )

    test_dataset = SyntheticClassificationDataset(
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

    print(
        "\n"
        "======================================="
    )
    print(
        "CUSTOM VISION TRANSFORMER"
    )
    print(
        "======================================="
    )

    custom_vit = VisionTransformer(
        image_size=64,
        patch_size=16,
        in_channels=3,
        num_classes=3,
        embed_dim=128,
        depth=4,
        num_heads=4,
        mlp_dim=256,
        dropout=0.1
    ).to(device)

    print(custom_vit)
    print(
        "Trainable parameters:",
        count_parameters(custom_vit)
    )

    train_model(
        model=custom_vit,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        epochs=5,
        lr=3e-4
    )

    probabilities, predicted_class = inference(
        model=custom_vit,
        image=image,
        device=device
    )

    print(
        "\nCustom ViT inference"
    )
    print(
        "Image shape:",
        image.shape
    )
    print(
        "True class:",
        int(true_label)
    )
    print(
        "Predicted class:",
        predicted_class
    )
    print(
        "Probabilities:",
        probabilities
    )

    criterion = nn.CrossEntropyLoss()

    custom_test_loss, custom_test_accuracy = evaluate(
        model=custom_vit,
        dataloader=test_loader,
        criterion=criterion,
        device=device
    )

    print(
        "\n"
        "======================================="
    )
    print(
        "TORCHVISION ViT-B/16"
    )
    print(
        "======================================="
    )

    framework_vit = vit_b_16(
        weights=None,
        image_size=64,
        num_classes=3
    ).to(device)

    print(framework_vit)
    print(
        "Trainable parameters:",
        count_parameters(framework_vit)
    )

    train_model(
        model=framework_vit,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        epochs=2,
        lr=1e-4
    )

    probabilities, predicted_class = inference(
        model=framework_vit,
        image=image,
        device=device
    )

    print(
        "\nTorchvision ViT-B/16 inference"
    )
    print(
        "Image shape:",
        image.shape
    )
    print(
        "True class:",
        int(true_label)
    )
    print(
        "Predicted class:",
        predicted_class
    )
    print(
        "Probabilities:",
        probabilities
    )

    framework_test_loss, framework_test_accuracy = evaluate(
        model=framework_vit,
        dataloader=test_loader,
        criterion=criterion,
        device=device
    )

    print(
        "\n"
        "======================================="
    )
    print(
        "COMPARISON"
    )
    print(
        "======================================="
    )

    print(
        f"Custom ViT loss:        {custom_test_loss:.4f}"
    )
    print(
        f"Torchvision ViT loss:   {framework_test_loss:.4f}"
    )
    print()
    print(
        f"Custom ViT accuracy:    {custom_test_accuracy:.4f}"
    )
    print(
        f"Torchvision accuracy:   {framework_test_accuracy:.4f}"
    )


if __name__ == "__main__":
    main()
