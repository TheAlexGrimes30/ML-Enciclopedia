import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import segmentation_models_pytorch as smp

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

class DoubleConv(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int
    ):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class DownBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int
    ):
        super().__init__()

        self.block = nn.Sequential(

            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),

            DoubleConv(
                in_channels=in_channels,
                out_channels=out_channels
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class UpBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            skip_channels: int,
            out_channels: int
    ):
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=2,
            stride=2
        )

        self.conv = DoubleConv(
            in_channels=(out_channels + skip_channels),
            out_channels=out_channels
        )

    def forward(
            self,
            x: torch.Tensor,
            skip: torch.Tensor
    ):
        x = self.up(x)
        x = torch.cat((skip, x), dim=1)
        x = self.conv(x)
        return x

class UNet(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            num_classes: int = 1
    ):
        super().__init__()

        self.encoder1 = DoubleConv(
            in_channels=in_channels,
            out_channels=32
        )

        self.encoder2 = DownBlock(
            in_channels=32,
            out_channels=64
        )

        self.encoder3 = DownBlock(
            in_channels=64,
            out_channels=128
        )

        self.bottleneck = DownBlock(
            in_channels=128,
            out_channels=256
        )

        self.decoder3 = UpBlock(
            in_channels=256,
            skip_channels=128,
            out_channels=128
        )

        self.decoder2 = UpBlock(
            in_channels=128,
            skip_channels=64,
            out_channels=64
        )

        self.decoder1 = UpBlock(
            in_channels=64,
            skip_channels=32,
            out_channels=32
        )

        self.segmentation_head = nn.Conv2d(
            in_channels=32,
            out_channels=num_classes,
            kernel_size=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        skip1 = self.encoder1(x)
        skip2 = self.encoder2(skip1)
        skip3 = self.encoder3(skip2)

        x = self.bottleneck(skip3)

        x = self.decoder3(x, skip3)
        x = self.decoder2(x, skip2)
        x = self.decoder1(x, skip1)

        logits = self.segmentation_head(x)

        return logits

class SyntheticSegmentationDataset(Dataset):

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
        masks = []

        for _ in range(
                n_samples
        ):

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

            y2 = (
                    y1
                    + rect_height
            )

            x2 = (
                    x1
                    + rect_width
            )

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

            image = image.clamp(
                0,
                1
            )

            images.append(
                image
            )

            masks.append(
                mask
            )

        self.images = torch.stack(
            images
        )

        self.masks = torch.stack(
            masks
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
            self.masks[index]
        )

def dice_score(
        logits: torch.Tensor,
        masks: torch.Tensor,
        threshold: float = 0.5,
        eps: float = 1e-7
):

    probabilities = torch.sigmoid(
        logits
    )

    predictions = (
            probabilities
            >= threshold
    ).float()

    intersection = (
            predictions
            * masks
    ).sum(
        dim=(1, 2, 3)
    )

    union = (
            predictions.sum(
                dim=(1, 2, 3)
            )
            + masks.sum(
                dim=(1, 2, 3)
            )
    )

    dice = (
            (
                    2.0
                    * intersection
                    + eps
            )
            /
            (
                    union
                    + eps
            )
    )

    return dice.mean().item()

def iou_score(
        logits: torch.Tensor,
        masks: torch.Tensor,
        threshold: float = 0.5,
        eps: float = 1e-7
):

    probabilities = torch.sigmoid(
        logits
    )

    predictions = (
            probabilities
            >= threshold
    ).float()

    intersection = (
            predictions
            * masks
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
                    intersection
                    + eps
            )
            /
            (
                    union
                    + eps
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

    for images, masks in tqdm(
            dataloader,
            desc="Training"
    ):

        images = images.to(
            device
        )

        masks = masks.to(
            device
        )

        optimizer.zero_grad()

        logits = model(
            images
        )

        loss = criterion(
            logits,
            masks
        )

        loss.backward()

        optimizer.step()

        batch_size = images.size(
            0
        )

        total_loss += (
                loss.item()
                * batch_size
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

    for images, masks in tqdm(
            dataloader,
            desc="Validation"
    ):

        images = images.to(
            device
        )

        masks = masks.to(
            device
        )

        logits = model(
            images
        )

        loss = criterion(
            logits,
            masks
        )

        batch_size = images.size(
            0
        )

        total_loss += (
                loss.item()
                * batch_size
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

    for epoch in range(
            epochs
    ):

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
        device,
        threshold: float = 0.5
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

    probabilities = torch.sigmoid(
        logits
    )

    predicted_mask = (
            probabilities
            >= threshold
    ).float()

    return (
        probabilities.squeeze(0).cpu(),
        predicted_mask.squeeze(0).cpu()
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

    train_dataset = SyntheticSegmentationDataset(
        n_samples=600,
        image_size=64,
        seed=42
    )

    test_dataset = SyntheticSegmentationDataset(
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

    image, true_mask = test_dataset[0]

    print(
        "\n"
        "======================================="
    )

    print(
        "CUSTOM SIMPLE U-NET"
    )

    print(
        "======================================="
    )

    custom_unet = UNet(
        in_channels=3,
        num_classes=1
    ).to(
        device
    )

    print(
        custom_unet
    )

    train_model(
        model=custom_unet,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        epochs=5,
        lr=1e-3
    )

    probabilities, predicted_mask = inference(
        model=custom_unet,
        image=image,
        device=device
    )

    print(
        "\nCustom U-Net inference"
    )

    print(
        "Image shape:",
        image.shape
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

    custom_logits = custom_unet(
        image.unsqueeze(0).to(device)
    )

    custom_dice = dice_score(
        custom_logits,
        true_mask.unsqueeze(0).to(device)
    )

    custom_iou = iou_score(
        custom_logits,
        true_mask.unsqueeze(0).to(device)
    )

    print(
        "Custom Dice:",
        round(
            custom_dice,
            4
        )
    )

    print(
        "Custom IoU:",
        round(
            custom_iou,
            4
        )
    )

    print(
        "\n"
        "======================================="
    )

    print(
        "SEGMENTATION MODELS PYTORCH U-NET"
    )

    print(
        "======================================="
    )

    framework_unet = smp.Unet(
        encoder_name="resnet18",
        encoder_weights=None,
        in_channels=3,
        classes=1
    ).to(
        device
    )

    print(
        framework_unet
    )

    train_model(
        model=framework_unet,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        epochs=5,
        lr=1e-3
    )

    probabilities, predicted_mask = inference(
        model=framework_unet,
        image=image,
        device=device
    )

    print(
        "\nFramework U-Net inference"
    )

    print(
        "Image shape:",
        image.shape
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

    framework_logits = framework_unet(
        image.unsqueeze(0).to(device)
    )

    framework_dice = dice_score(
        framework_logits,
        true_mask.unsqueeze(0).to(device)
    )

    framework_iou = iou_score(
        framework_logits,
        true_mask.unsqueeze(0).to(device)
    )

    print(
        "Framework Dice:",
        round(
            framework_dice,
            4
        )
    )

    print(
        "Framework IoU:",
        round(
            framework_iou,
            4
        )
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
        f"Custom U-Net Dice:    {custom_dice:.4f}"
    )

    print(
        f"Framework U-Net Dice: {framework_dice:.4f}"
    )

    print()

    print(
        f"Custom U-Net IoU:     {custom_iou:.4f}"
    )

    print(
        f"Framework U-Net IoU:  {framework_iou:.4f}"
    )


if __name__ == "__main__":
    main()