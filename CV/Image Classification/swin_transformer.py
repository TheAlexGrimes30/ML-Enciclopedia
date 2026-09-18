import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import swin_t
from tqdm import tqdm

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

class PatchEmbedding(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            embed_dim: int = 64,
            patch_size: int = 4
    ):
        super().__init__()

        self.projection = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projection(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)

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


class DropPath(nn.Module):
    def __init__(
            self,
            drop_probability: float = 0.0
    ):
        super().__init__()
        self.drop_probability = drop_probability

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_probability or not self.training:
            return x

        keep_probability = 1.0 - self.drop_probability

        shape = x.shape[0], *([1] * (x.ndim - 1))

        random_tensor = keep_probability + torch.rand(
            shape,
            dtype=x.dtype,
            device=x.device
        )

        random_tensor.floor_()

        return x / keep_probability * random_tensor

def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    batch_size, height, width, channels = x.shape

    x = x.view(
        batch_size,
        height // window_size,
        window_size,
        width // window_size,
        window_size,
        channels
    )

    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = windows.view(-1, window_size * window_size, channels)

    return windows


def window_reverse(
        windows: torch.Tensor,
        window_size: int,
        height: int,
        width: int,
        batch_size: int
) -> torch.Tensor:
    channels = windows.shape[-1]

    x = windows.view(
        batch_size,
        height // window_size,
        width // window_size,
        window_size,
        window_size,
        channels
    )

    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()

    x = x.view(
        batch_size,
        height,
        width,
        channels
    )

    return x

class WindowAttention(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            window_size: int,
            num_heads: int,
            dropout: float = 0.0
    ):
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(
                "embed_dim must be divisible by num_heads"
            )

        self.embed_dim = embed_dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attention_dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(embed_dim, embed_dim)
        self.projection_dropout = nn.Dropout(dropout)

        number_of_relative_positions = (
                (2 * window_size - 1)
                * (2 * window_size - 1)
        )

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                number_of_relative_positions,
                num_heads
            )
        )

        coordinates = torch.stack(
            torch.meshgrid(
                torch.arange(window_size),
                torch.arange(window_size),
                indexing="ij"
            )
        )

        coordinates_flatten = coordinates.flatten(start_dim=1)

        relative_coordinates = (
                coordinates_flatten[:, :, None]
                - coordinates_flatten[:, None, :]
        )

        relative_coordinates = relative_coordinates.permute(
            1,
            2,
            0
        ).contiguous()

        relative_coordinates[:, :, 0] += window_size - 1
        relative_coordinates[:, :, 1] += window_size - 1
        relative_coordinates[:, :, 0] *= 2 * window_size - 1

        relative_position_index = relative_coordinates.sum(-1)

        self.register_buffer(
            "relative_position_index",
            relative_position_index
        )

        nn.init.trunc_normal_(
            self.relative_position_bias_table,
            std=0.02
        )

    def forward(
            self,
            x: torch.Tensor,
            attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        batch_windows, num_tokens, channels = x.shape

        qkv = self.qkv(x)

        qkv = qkv.reshape(
            batch_windows,
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

        query, key, value = qkv.unbind(dim=0)

        query = query * self.scale

        attention_scores = (
                query
                @ key.transpose(-2, -1)
        )

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.reshape(-1)
        ]

        relative_position_bias = relative_position_bias.view(
            num_tokens,
            num_tokens,
            self.num_heads
        )

        relative_position_bias = relative_position_bias.permute(
            2,
            0,
            1
        ).contiguous()

        attention_scores = (
                attention_scores
                + relative_position_bias.unsqueeze(0)
        )

        if attention_mask is not None:
            number_of_windows = attention_mask.shape[0]

            attention_scores = attention_scores.view(
                batch_windows // number_of_windows,
                number_of_windows,
                self.num_heads,
                num_tokens,
                num_tokens
            )

            attention_scores = (
                    attention_scores
                    + attention_mask.unsqueeze(0).unsqueeze(2)
            )

            attention_scores = attention_scores.view(
                -1,
                self.num_heads,
                num_tokens,
                num_tokens
            )

        attention = torch.softmax(
            attention_scores,
            dim=-1
        )

        attention = self.attention_dropout(attention)

        x = attention @ value

        x = x.transpose(
            1,
            2
        ).reshape(
            batch_windows,
            num_tokens,
            channels
        )

        x = self.projection(x)
        x = self.projection_dropout(x)

        return x

class SwinTransformerBlock(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            window_size: int = 4,
            shift_size: int = 0,
            mlp_ratio: float = 4.0,
            dropout: float = 0.0,
            drop_path: float = 0.0
    ):
        super().__init__()

        if shift_size >= window_size:
            raise ValueError(
                "shift_size must be smaller than window_size"
            )

        self.embed_dim = embed_dim
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(embed_dim)

        self.attention = WindowAttention(
            embed_dim=embed_dim,
            window_size=window_size,
            num_heads=num_heads,
            dropout=dropout
        )

        self.drop_path = DropPath(drop_path)

        self.norm2 = nn.LayerNorm(embed_dim)

        self.mlp = MLP(
            embed_dim=embed_dim,
            mlp_dim=int(embed_dim * mlp_ratio),
            dropout=dropout
        )

    def _create_attention_mask(
            self,
            height: int,
            width: int,
            device
    ) -> torch.Tensor:
        image_mask = torch.zeros(
            1,
            height,
            width,
            1,
            device=device
        )

        height_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None)
        )

        width_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None)
        )

        counter = 0

        for height_slice in height_slices:
            for width_slice in width_slices:
                image_mask[
                    :,
                    height_slice,
                    width_slice,
                    :
                ] = counter

                counter += 1

        mask_windows = window_partition(
            image_mask,
            self.window_size
        ).squeeze(-1)

        attention_mask = (
                mask_windows.unsqueeze(1)
                - mask_windows.unsqueeze(2)
        )

        attention_mask = attention_mask.masked_fill(
            attention_mask != 0,
            float("-100.0")
        )

        attention_mask = attention_mask.masked_fill(
            attention_mask == 0,
            float("0.0")
        )

        return attention_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, height, width, channels = x.shape

        if (
                height % self.window_size != 0
                or width % self.window_size != 0
        ):
            raise ValueError(
                "Feature-map height and width must be divisible "
                "by window_size in this educational implementation"
            )

        shortcut = x
        x = self.norm1(x)

        if self.shift_size > 0:
            shifted_x = torch.roll(
                x,
                shifts=(-self.shift_size, -self.shift_size),
                dims=(1, 2)
            )

            attention_mask = self._create_attention_mask(
                height=height,
                width=width,
                device=x.device
            )
        else:
            shifted_x = x
            attention_mask = None

        windows = window_partition(
            shifted_x,
            self.window_size
        )

        attention_windows = self.attention(
            windows,
            attention_mask=attention_mask
        )

        shifted_x = window_reverse(
            windows=attention_windows,
            window_size=self.window_size,
            height=height,
            width=width,
            batch_size=batch_size
        )

        if self.shift_size > 0:
            x = torch.roll(
                shifted_x,
                shifts=(self.shift_size, self.shift_size),
                dims=(1, 2)
            )
        else:
            x = shifted_x

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(
            self.mlp(
                self.norm2(x)
            )
        )

        return x

class PatchMerging(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()

        self.norm = nn.LayerNorm(4 * embed_dim)

        self.reduction = nn.Linear(
            4 * embed_dim,
            2 * embed_dim,
            bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, height, width, _ = x.shape

        if height % 2 != 0 or width % 2 != 0:
            raise ValueError(
                "Feature-map height and width must be even "
                "before PatchMerging"
            )

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]

        x = torch.cat(
            [x0, x1, x2, x3],
            dim=-1
        )

        x = self.norm(x)
        x = self.reduction(x)

        return x


class SwinStage(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            depth: int,
            num_heads: int,
            window_size: int,
            mlp_ratio: float = 4.0,
            dropout: float = 0.0,
            drop_path_rates: list[float] | None = None,
            downsample: bool = True
    ):
        super().__init__()

        if drop_path_rates is None:
            drop_path_rates = [0.0] * depth

        blocks = []

        for block_index in range(depth):
            shift_size = (
                0
                if block_index % 2 == 0
                else window_size // 2
            )

            blocks.append(
                SwinTransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=shift_size,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    drop_path=drop_path_rates[block_index]
                )
            )

        self.blocks = nn.Sequential(*blocks)

        self.downsample = (
            PatchMerging(embed_dim)
            if downsample
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(x)

        if self.downsample is not None:
            x = self.downsample(x)

        return x

class SwinTransformer(nn.Module):
    def __init__(
            self,
            image_size: int = 64,
            patch_size: int = 4,
            in_channels: int = 3,
            num_classes: int = 3,
            embed_dim: int = 64,
            depths: tuple[int, ...] = (2, 2, 2, 2),
            num_heads: tuple[int, ...] = (2, 4, 8, 16),
            window_size: int = 4,
            mlp_ratio: float = 4.0,
            dropout: float = 0.1,
            drop_path_rate: float = 0.1
    ):
        super().__init__()

        if len(depths) != len(num_heads):
            raise ValueError(
                "depths and num_heads must have the same length"
            )

        number_of_stages = len(depths)

        minimum_divisor = (
                patch_size
                * (2 ** (number_of_stages - 1))
        )

        if image_size % minimum_divisor != 0:
            raise ValueError(
                "image_size must be divisible by "
                f"{minimum_divisor} for this configuration"
            )

        self.patch_embedding = PatchEmbedding(
            in_channels=in_channels,
            embed_dim=embed_dim,
            patch_size=patch_size
        )

        self.embedding_dropout = nn.Dropout(dropout)

        total_depth = sum(depths)

        drop_path_rates = torch.linspace(
            0,
            drop_path_rate,
            total_depth
        ).tolist()

        stages = []
        depth_offset = 0

        for stage_index in range(number_of_stages):
            stage_embed_dim = (
                    embed_dim
                    * (2 ** stage_index)
            )

            stage_depth = depths[stage_index]

            stage_drop_path_rates = drop_path_rates[
                depth_offset:
                depth_offset + stage_depth
            ]

            stages.append(
                SwinStage(
                    embed_dim=stage_embed_dim,
                    depth=stage_depth,
                    num_heads=num_heads[stage_index],
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    drop_path_rates=stage_drop_path_rates,
                    downsample=(stage_index < number_of_stages - 1)
                )
            )

            depth_offset += stage_depth

        self.stages = nn.ModuleList(stages)

        final_embed_dim = (
                embed_dim
                * (2 ** (number_of_stages - 1))
        )

        self.norm = nn.LayerNorm(final_embed_dim)

        self.classifier = nn.Linear(
            final_embed_dim,
            num_classes
        )

        self.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(
                module.weight,
                std=0.02
            )

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embedding(x)
        x = self.embedding_dropout(x)

        for stage in self.stages:
            x = stage(x)

        x = self.norm(x)

        x = x.mean(
            dim=(1, 2)
        )

        logits = self.classifier(x)

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
            raise ValueError(
                "This synthetic dataset is designed for 3 classes"
            )

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
        "CUSTOM SWIN TRANSFORMER"
    )
    print(
        "======================================="
    )

    custom_swin = SwinTransformer(
        image_size=64,
        patch_size=4,
        in_channels=3,
        num_classes=3,
        embed_dim=64,
        depths=(2, 2, 2, 2),
        num_heads=(2, 4, 8, 16),
        window_size=2,
        mlp_ratio=4.0,
        dropout=0.1,
        drop_path_rate=0.1
    ).to(device)

    print(custom_swin)
    print(
        "Trainable parameters:",
        count_parameters(custom_swin)
    )

    train_model(
        model=custom_swin,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        epochs=5,
        lr=3e-4
    )

    probabilities, predicted_class = inference(
        model=custom_swin,
        image=image,
        device=device
    )

    print(
        "\nCustom Swin inference"
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
        model=custom_swin,
        dataloader=test_loader,
        criterion=criterion,
        device=device
    )

    print(
        "\n"
        "======================================="
    )
    print(
        "TORCHVISION SWIN-T"
    )
    print(
        "======================================="
    )

    framework_swin = swin_t(
        weights=None,
        num_classes=3
    ).to(device)

    print(framework_swin)
    print(
        "Trainable parameters:",
        count_parameters(framework_swin)
    )

    train_model(
        model=framework_swin,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        epochs=2,
        lr=1e-4
    )

    probabilities, predicted_class = inference(
        model=framework_swin,
        image=image,
        device=device
    )

    print(
        "\nTorchvision Swin-T inference"
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
        model=framework_swin,
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
        f"Custom Swin loss:       {custom_test_loss:.4f}"
    )
    print(
        f"Torchvision Swin loss:  {framework_test_loss:.4f}"
    )
    print()
    print(
        f"Custom Swin accuracy:   {custom_test_accuracy:.4f}"
    )
    print(
        f"Torchvision accuracy:   {framework_test_accuracy:.4f}"
    )


if __name__ == "__main__":
    main()
