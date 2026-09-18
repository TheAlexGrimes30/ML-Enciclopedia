import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

class PatchEmbedding(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            embed_dim: int = 128,
            patch_size: int = 4
    ):
        super().__init__()

        self.projection = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)

class TransformerBlock(nn.Module):
    def  __init__(
            self,
            embed_dim: int = 128,
            num_heads: int = 4,
            mlp_ratio: int = 4
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)

        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.norm2 = nn.LayerNorm(embed_dim)

        hidden_dim = embed_dim * mlp_ratio

        self.mlp = nn.Sequential(
            nn.Linear(
                embed_dim,
                hidden_dim
            ),

            nn.GELU(),

            nn.Linear(
                hidden_dim,
                embed_dim
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(x)

        attention_output, _ = self.attention(
            normalized,
            normalized,
            normalized
        )

        x = x + attention_output
        x = x + self.mlp(self.norm2(x))

        return x

class ImageEncoder(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            embed_dim: int = 128,
            patch_size: int = 4,
            depth: int = 4,
            num_heads: int =4
    ):
        super().__init__()

        self.patching_embedding = PatchEmbedding(
            in_channels=in_channels,
            embed_dim=embed_dim,
            patch_size=patch_size
        )

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads
                )
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patching_embedding(x)
        batch_size, channels, height, width = x.shape

        x = x.flatten(start_dim=2)
        x = x.transpose(1, 2)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        x = x.transpose(1, 2)
        x = x.reshape(
            batch_size,
            channels,
            height,
            width
        )

        return x

class PointPromptEncoder(nn.Module):
    def __init__(
            self,
            embed_dim: int = 128
    ):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, point: torch.Tensor) -> torch.Tensor:
        return self.mlp(point)

class MaskDecoder(nn.Module):
    def __init__(
            self,
            embed_dim: int = 128
    ):
        super().__init__()

        self.decoder = nn.Sequential(

            nn.Conv2d(
                in_channels=embed_dim,
                out_channels=128,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=64,
                kernel_size=2,
                stride=2
            ),

            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=32,
                kernel_size=2,
                stride=2
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                in_channels=32,
                out_channels=1,
                kernel_size=1
            )
        )

    def forward(
            self,
            image_embeddings: torch.Tensor,
            prompt_embeddings: torch.Tensor
    ) -> torch.Tensor:

        prompt_embeddings = (
            prompt_embeddings
            .unsqueeze(dim=-1)
            .unsqueeze(dim=-1)
        )

        x = image_embeddings + prompt_embeddings
        mask_logits = self.decoder(x)

        return mask_logits

class SAM(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            embed_dim: int = 128,
            patch_size: int = 4,
            depth: int = 4,
            num_heads: int = 4
    ):
        super().__init__()

        self.image_encoder = ImageEncoder(
            in_channels=in_channels,
            embed_dim=embed_dim,
            patch_size=patch_size,
            depth=depth,
            num_heads=num_heads
        )

        self.prompt_encoder = PointPromptEncoder(embed_dim=embed_dim)
        self.mask_decoder = MaskDecoder(embed_dim=embed_dim)

    def forward(
            self,
            image: torch.Tensor,
            point: torch.Tensor
    ) -> torch.Tensor:

        image_embeddings = self.image_encoder(image)
        prompt_embedding = self.prompt_encoder(point)

        mask_logits = self.mask_decoder(
            image_embeddings,
            prompt_embedding
        )

        return mask_logits

class SyntheticPromptSegmentationDataset(Dataset):

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
        points = []
        masks = []

        for _ in range(n_samples):

            image = torch.randn(
                3,
                image_size,
                image_size,
                generator=generator
            ) * 0.08

            mask = torch.zeros(
                1,
                image_size,
                image_size
            )

            rect_height = int(
                torch.randint(
                    low=image_size // 6,
                    high=image_size // 2,
                    size=(1,),
                    generator=generator
                ).item()
            )

            rect_width = int(
                torch.randint(
                    low=image_size // 6,
                    high=image_size // 2,
                    size=(1,),
                    generator=generator
                ).item()
            )

            y1 = int(
                torch.randint(
                    low=0,
                    high=image_size - rect_height,
                    size=(1,),
                    generator=generator
                ).item()
            )

            x1 = int(
                torch.randint(
                    low=0,
                    high=image_size - rect_width,
                    size=(1,),
                    generator=generator
                ).item()
            )

            y2 = y1 + rect_height
            x2 = x1 + rect_width

            mask[
                :,
                y1:y2,
                x1:x2
            ] = 1.0

            image[
                :,
                y1:y2,
                x1:x2
            ] += 1.0

            point_y = int(
                torch.randint(
                    low=y1,
                    high=y2,
                    size=(1,),
                    generator=generator
                ).item()
            )

            point_x = int(
                torch.randint(
                    low=x1,
                    high=x2,
                    size=(1,),
                    generator=generator
                ).item()
            )

            point = torch.tensor(
                [
                    point_x / (image_size - 1),
                    point_y / (image_size - 1)
                ],
                dtype=torch.float32
            )

            image = image.clamp(
                0,
                1
            )

            images.append(image)
            points.append(point)
            masks.append(mask)

        self.images = torch.stack(images)
        self.points = torch.stack(points)
        self.masks = torch.stack(masks)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):

        return (
            self.images[index],
            self.points[index],
            self.masks[index]
        )

def dice_score(
        logits: torch.Tensor,
        masks: torch.Tensor,
        threshold: float = 0.5,
        eps: float = 1e-7
):
    probabilities = torch.sigmoid(logits)
    predictions = (probabilities >= threshold).float()

    intersection = (predictions * masks).sum(dim=(1, 2, 3))

    union = (
            predictions.sum(
                dim=(1, 2, 3)
            )
            + masks.sum(
        dim=(1, 2, 3)
    )
    )

    dice = ((2.0 * intersection + eps) /(union + eps))

    return dice.mean().item()

def iou_score(
        logits: torch.Tensor,
        masks: torch.Tensor,
        threshold: float = 0.5,
        eps: float = 1e-7
):

    probabilities = torch.sigmoid(logits)

    predictions = (
            probabilities >= threshold
    ).float()

    intersection = (
            predictions * masks
    ).sum(
        dim=(1, 2, 3)
    )

    union = (
            predictions
            + masks
            - predictions * masks
    ).sum(
        dim=(1, 2, 3)
    )

    iou = (
            (
                    intersection + eps
            )
            /
            (
                    union + eps
            )
    )

    return iou.mean().item()

def train_epoch(
        model: nn.Module,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion,
        device
):

    model.train()

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total = 0

    for images, points, masks in tqdm(
            dataloader,
            desc="Training"
    ):

        images = images.to(device)
        points = points.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        logits = model(
            images,
            points
        )

        loss = criterion(
            logits,
            masks
        )

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)

        total_loss += (
                loss.item() * batch_size
        )

        total_dice += (
                dice_score(
                    logits,
                    masks
                )
                * batch_size
        )

        total_iou += (
                iou_score(
                    logits,
                    masks
                )
                * batch_size
        )

        total += batch_size

    return (
        total_loss / total,
        total_dice / total,
        total_iou / total
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
    total_dice = 0.0
    total_iou = 0.0
    total = 0

    for images, points, masks in tqdm(
            dataloader,
            desc="Validation"
    ):

        images = images.to(device)
        points = points.to(device)
        masks = masks.to(device)

        logits = model(
            images,
            points
        )

        loss = criterion(
            logits,
            masks
        )

        batch_size = images.size(0)

        total_loss += (
                loss.item() * batch_size
        )

        total_dice += (
                dice_score(
                    logits,
                    masks
                )
                * batch_size
        )

        total_iou += (
                iou_score(
                    logits,
                    masks
                )
                * batch_size
        )

        total += batch_size

    return (
        total_loss / total,
        total_dice / total,
        total_iou / total
    )


def train_model(
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        device,
        epochs: int = 5,
        lr: float = 1e-3
):

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr
    )

    for epoch in range(epochs):

        train_loss, train_dice, train_iou = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device
        )

        test_loss, test_dice, test_iou = evaluate(
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
            f"Train Dice: {train_dice:.4f}"
        )

        print(
            f"Train IoU: {train_iou:.4f}"
        )

        print(
            f"Test loss: {test_loss:.4f}"
        )

        print(
            f"Test Dice: {test_dice:.4f}"
        )

        print(
            f"Test IoU: {test_iou:.4f}"
        )

        print()


@torch.no_grad()
def inference(
        model: nn.Module,
        image: torch.Tensor,
        point: torch.Tensor,
        device,
        threshold: float = 0.5
):

    model.eval()

    image = image.unsqueeze(
        dim=0
    ).to(device)

    point = point.unsqueeze(
        dim=0
    ).to(device)

    logits = model(
        image,
        point
    )

    probabilities = torch.sigmoid(
        logits
    )

    predicted_mask = (
            probabilities >= threshold
    ).float()

    return (
        probabilities.squeeze(0).cpu(),
        predicted_mask.squeeze(0).cpu()
    )


def count_parameters(
        model: nn.Module
):

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

    train_dataset = SyntheticPromptSegmentationDataset(
        n_samples=600,
        image_size=64,
        seed=42
    )

    test_dataset = SyntheticPromptSegmentationDataset(
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
        "CUSTOM SIMPLE SAM"
    )

    print(
        "======================================="
    )

    model = SAM(
        in_channels=3,
        embed_dim=128,
        patch_size=4,
        depth=4,
        num_heads=4
    ).to(device)

    print(model)

    print(
        "Trainable parameters:",
        count_parameters(model)
    )

    train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        epochs=5,
        lr=1e-3
    )

    image, point, true_mask = (
        test_dataset[0]
    )

    probabilities, predicted_mask = inference(
        model=model,
        image=image,
        point=point,
        device=device
    )

    print(
        "\nSimple SAM inference"
    )

    print(
        "Image shape:",
        image.shape
    )

    print(
        "Point prompt:",
        point
    )

    print(
        "True mask shape:",
        true_mask.shape
    )

    print(
        "Predicted mask shape:",
        predicted_mask.shape
    )

    print(
        "True foreground pixels:",
        int(
            true_mask.sum().item()
        )
    )

    print(
        "Predicted foreground pixels:",
        int(
            predicted_mask.sum().item()
        )
    )

    sample_logits = model(
        image.unsqueeze(0).to(device),
        point.unsqueeze(0).to(device)
    )

    sample_dice = dice_score(
        sample_logits,
        true_mask.unsqueeze(0).to(device)
    )

    sample_iou = iou_score(
        sample_logits,
        true_mask.unsqueeze(0).to(device)
    )

    print(
        "Sample Dice:",
        round(
            sample_dice,
            4
        )
    )

    print(
        "Sample IoU:",
        round(
            sample_iou,
            4
        )
    )


if __name__ == "__main__":
    main()