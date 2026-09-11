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


class CNNBackbone(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            hidden_dim: int = 128
    ):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                in_channels=64,
                out_channels=hidden_dim,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class PositionalEmbedding(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 128,
            max_height: int = 64,
            max_width: int = 64
    ):
        super().__init__()

        self.row_embedding = nn.Embedding(
            max_height,
            hidden_dim // 2
        )

        self.column_embedding = nn.Embedding(
            max_width,
            hidden_dim // 2
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = x.shape

        rows = self.row_embedding(
            torch.arange(
                height,
                device=x.device
            )
        )

        columns = self.column_embedding(
            torch.arange(
                width,
                device=x.device
            )
        )

        position = torch.cat(
            (
                columns.unsqueeze(0).expand(
                    height,
                    -1,
                    -1
                ),

                rows.unsqueeze(1).expand(
                    -1,
                    width,
                    -1
                )
            ),
            dim=-1
        )

        position = position.reshape(
            1,
            height * width,
            channels
        )

        return position.expand(
            batch_size,
            -1,
            -1
        )


class TransformerEncoderBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 128,
            num_heads: int = 4,
            mlp_ratio: int = 4
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(hidden_dim)

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.norm2 = nn.LayerNorm(hidden_dim)

        mlp_dim = hidden_dim * mlp_ratio

        self.mlp = nn.Sequential(
            nn.Linear(
                hidden_dim,
                mlp_dim
            ),

            nn.ReLU(inplace=True),

            nn.Linear(
                mlp_dim,
                hidden_dim
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(x)

        attention_output, _ = self.attention(
            normalized,
            normalized,
            normalized,
            need_weights=False
        )

        x = x + attention_output
        x = x + self.mlp(self.norm2(x))

        return x


class TransformerDecoderBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 128,
            num_heads: int = 4,
            mlp_ratio: int = 4
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(hidden_dim)

        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.norm2 = nn.LayerNorm(hidden_dim)

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.norm3 = nn.LayerNorm(hidden_dim)

        mlp_dim = hidden_dim * mlp_ratio

        self.mlp = nn.Sequential(
            nn.Linear(
                hidden_dim,
                mlp_dim
            ),

            nn.ReLU(inplace=True),

            nn.Linear(
                mlp_dim,
                hidden_dim
            )
        )

    def forward(
            self,
            queries: torch.Tensor,
            memory: torch.Tensor
    ) -> torch.Tensor:

        normalized_queries = self.norm1(queries)

        self_attention_output, _ = self.self_attention(
            normalized_queries,
            normalized_queries,
            normalized_queries,
            need_weights=False
        )

        queries = queries + self_attention_output

        normalized_queries = self.norm2(queries)

        cross_attention_output, _ = self.cross_attention(
            normalized_queries,
            memory,
            memory,
            need_weights=False
        )

        queries = queries + cross_attention_output
        queries = queries + self.mlp(self.norm3(queries))

        return queries


class BoxHead(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 128
    ):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(
                hidden_dim,
                hidden_dim
            ),

            nn.ReLU(inplace=True),

            nn.Linear(
                hidden_dim,
                hidden_dim
            ),

            nn.ReLU(inplace=True),

            nn.Linear(
                hidden_dim,
                4
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.mlp(x))


class DETR(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            num_classes: int = 2,
            hidden_dim: int = 128,
            num_queries: int = 5,
            encoder_depth: int = 3,
            decoder_depth: int = 3,
            num_heads: int = 4
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_queries = num_queries

        self.backbone = CNNBackbone(
            in_channels=in_channels,
            hidden_dim=hidden_dim
        )

        self.position_embedding = PositionalEmbedding(
            hidden_dim=hidden_dim
        )

        self.encoder_blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads
                )
                for _ in range(encoder_depth)
            ]
        )

        self.object_queries = nn.Embedding(
            num_embeddings=num_queries,
            embedding_dim=hidden_dim
        )

        self.decoder_blocks = nn.ModuleList(
            [
                TransformerDecoderBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads
                )
                for _ in range(decoder_depth)
            ]
        )

        self.classification_head = nn.Linear(
            hidden_dim,
            num_classes + 1
        )

        self.box_head = BoxHead(
            hidden_dim=hidden_dim
        )

    def forward(self, images: torch.Tensor):
        features = self.backbone(images)

        batch_size, channels, height, width = features.shape

        memory = features.flatten(
            start_dim=2
        )

        memory = memory.transpose(
            1,
            2
        )

        memory = memory + self.position_embedding(
            features
        )

        for block in self.encoder_blocks:
            memory = block(memory)

        queries = self.object_queries.weight.unsqueeze(
            dim=0
        )

        queries = queries.expand(
            batch_size,
            -1,
            -1
        )

        for block in self.decoder_blocks:
            queries = block(
                queries,
                memory
            )

        pred_logits = self.classification_head(
            queries
        )

        pred_boxes = self.box_head(
            queries
        )

        return (
            pred_logits,
            pred_boxes
        )


class SyntheticDetectionDataset(Dataset):
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
        boxes = []

        for _ in range(n_samples):
            image = torch.randn(
                3,
                image_size,
                image_size,
                generator=generator
            ) * 0.05

            label = int(
                torch.randint(
                    low=0,
                    high=2,
                    size=(1,),
                    generator=generator
                ).item()
            )

            object_height = int(
                torch.randint(
                    low=image_size // 6,
                    high=image_size // 2,
                    size=(1,),
                    generator=generator
                ).item()
            )

            object_width = int(
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
                    high=image_size - object_height,
                    size=(1,),
                    generator=generator
                ).item()
            )

            x1 = int(
                torch.randint(
                    low=0,
                    high=image_size - object_width,
                    size=(1,),
                    generator=generator
                ).item()
            )

            y2 = y1 + object_height
            x2 = x1 + object_width

            if label == 0:
                image[
                    0,
                    y1:y2,
                    x1:x2
                ] += 1.0
            else:
                image[
                    1,
                    y1:y2,
                    x1:x2
                ] += 1.0

            center_x = (
                (x1 + x2)
                / 2
                / image_size
            )

            center_y = (
                (y1 + y2)
                / 2
                / image_size
            )

            width = (
                (x2 - x1)
                / image_size
            )

            height = (
                (y2 - y1)
                / image_size
            )

            box = torch.tensor(
                [
                    center_x,
                    center_y,
                    width,
                    height
                ],
                dtype=torch.float32
            )

            image = image.clamp(
                0,
                1
            )

            images.append(image)
            labels.append(label)
            boxes.append(box)

        self.images = torch.stack(images)

        self.labels = torch.tensor(
            labels,
            dtype=torch.long
        )

        self.boxes = torch.stack(boxes)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        return (
            self.images[index],
            self.labels[index],
            self.boxes[index]
        )


def box_iou(
        predicted_boxes: torch.Tensor,
        true_boxes: torch.Tensor,
        eps: float = 1e-7
):
    pred_cx, pred_cy, pred_w, pred_h = predicted_boxes.unbind(
        dim=-1
    )

    true_cx, true_cy, true_w, true_h = true_boxes.unbind(
        dim=-1
    )

    pred_x1 = pred_cx - pred_w / 2
    pred_y1 = pred_cy - pred_h / 2
    pred_x2 = pred_cx + pred_w / 2
    pred_y2 = pred_cy + pred_h / 2

    true_x1 = true_cx - true_w / 2
    true_y1 = true_cy - true_h / 2
    true_x2 = true_cx + true_w / 2
    true_y2 = true_cy + true_h / 2

    intersection_x1 = torch.maximum(
        pred_x1,
        true_x1
    )

    intersection_y1 = torch.maximum(
        pred_y1,
        true_y1
    )

    intersection_x2 = torch.minimum(
        pred_x2,
        true_x2
    )

    intersection_y2 = torch.minimum(
        pred_y2,
        true_y2
    )

    intersection = (
        (intersection_x2 - intersection_x1).clamp(min=0)
        *
        (intersection_y2 - intersection_y1).clamp(min=0)
    )

    pred_area = pred_w * pred_h
    true_area = true_w * true_h

    union = (
        pred_area
        + true_area
        - intersection
    )

    return intersection / (union + eps)


def detr_loss(
        pred_logits: torch.Tensor,
        pred_boxes: torch.Tensor,
        labels: torch.Tensor,
        boxes: torch.Tensor,
        num_classes: int
):
    batch_size, num_queries, _ = pred_logits.shape

    probabilities = pred_logits.softmax(
        dim=-1
    )

    class_cost = torch.zeros(
        batch_size,
        num_queries,
        device=pred_logits.device
    )

    for batch_index in range(batch_size):
        class_cost[batch_index] = -probabilities[
            batch_index,
            :,
            labels[batch_index]
        ]

    box_cost = torch.abs(
        pred_boxes
        - boxes.unsqueeze(dim=1)
    ).sum(
        dim=-1
    )

    matching_cost = (
        class_cost
        + 5.0 * box_cost
    )

    best_queries = matching_cost.argmin(
        dim=1
    )

    class_targets = torch.full(
        (
            batch_size,
            num_queries
        ),
        fill_value=num_classes,
        dtype=torch.long,
        device=pred_logits.device
    )

    class_targets[
        torch.arange(
            batch_size,
            device=pred_logits.device
        ),
        best_queries
    ] = labels

    classification_loss = nn.functional.cross_entropy(
        pred_logits.reshape(
            -1,
            num_classes + 1
        ),
        class_targets.reshape(-1)
    )

    matched_boxes = pred_boxes[
        torch.arange(
            batch_size,
            device=pred_logits.device
        ),
        best_queries
    ]

    bbox_loss = nn.functional.l1_loss(
        matched_boxes,
        boxes
    )

    loss = (
        classification_loss
        + 5.0 * bbox_loss
    )

    return (
        loss,
        best_queries
    )


def detection_metrics(
        pred_logits: torch.Tensor,
        pred_boxes: torch.Tensor,
        labels: torch.Tensor,
        boxes: torch.Tensor,
        best_queries: torch.Tensor
):
    batch_indices = torch.arange(
        labels.size(0),
        device=labels.device
    )

    matched_logits = pred_logits[
        batch_indices,
        best_queries
    ]

    matched_boxes = pred_boxes[
        batch_indices,
        best_queries
    ]

    predicted_labels = matched_logits[
        :,
        :-1
    ].argmax(
        dim=-1
    )

    accuracy = (
        predicted_labels
        == labels
    ).float().mean().item()

    iou = box_iou(
        matched_boxes,
        boxes
    ).mean().item()

    return (
        accuracy,
        iou
    )


def train_epoch(
        model: nn.Module,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device
):
    model.train()

    total_loss = 0.0
    total_accuracy = 0.0
    total_iou = 0.0
    total = 0

    for images, labels, boxes in tqdm(
            dataloader,
            desc="Training"
    ):
        images = images.to(device)
        labels = labels.to(device)
        boxes = boxes.to(device)

        optimizer.zero_grad()

        pred_logits, pred_boxes = model(
            images
        )

        loss, best_queries = detr_loss(
            pred_logits=pred_logits,
            pred_boxes=pred_boxes,
            labels=labels,
            boxes=boxes,
            num_classes=model.num_classes
        )

        loss.backward()
        optimizer.step()

        accuracy, iou = detection_metrics(
            pred_logits=pred_logits,
            pred_boxes=pred_boxes,
            labels=labels,
            boxes=boxes,
            best_queries=best_queries
        )

        batch_size = images.size(0)

        total_loss += loss.item() * batch_size
        total_accuracy += accuracy * batch_size
        total_iou += iou * batch_size
        total += batch_size

    return (
        total_loss / total,
        total_accuracy / total,
        total_iou / total
    )


@torch.no_grad()
def evaluate(
        model: nn.Module,
        dataloader: DataLoader,
        device
):
    model.eval()

    total_loss = 0.0
    total_accuracy = 0.0
    total_iou = 0.0
    total = 0

    for images, labels, boxes in tqdm(
            dataloader,
            desc="Validation"
    ):
        images = images.to(device)
        labels = labels.to(device)
        boxes = boxes.to(device)

        pred_logits, pred_boxes = model(
            images
        )

        loss, best_queries = detr_loss(
            pred_logits=pred_logits,
            pred_boxes=pred_boxes,
            labels=labels,
            boxes=boxes,
            num_classes=model.num_classes
        )

        accuracy, iou = detection_metrics(
            pred_logits=pred_logits,
            pred_boxes=pred_boxes,
            labels=labels,
            boxes=boxes,
            best_queries=best_queries
        )

        batch_size = images.size(0)

        total_loss += loss.item() * batch_size
        total_accuracy += accuracy * batch_size
        total_iou += iou * batch_size
        total += batch_size

    return (
        total_loss / total,
        total_accuracy / total,
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
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr
    )

    for epoch in range(epochs):
        train_loss, train_accuracy, train_iou = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device
        )

        test_loss, test_accuracy, test_iou = evaluate(
            model=model,
            dataloader=test_loader,
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
            f"Train IoU: {train_iou:.4f}"
        )

        print(
            f"Test loss: {test_loss:.4f}"
        )

        print(
            f"Test accuracy: {test_accuracy:.4f}"
        )

        print(
            f"Test IoU: {test_iou:.4f}"
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
    ).to(device)

    pred_logits, pred_boxes = model(
        image
    )

    probabilities = pred_logits.softmax(
        dim=-1
    )[0]

    object_probabilities = probabilities[
        :,
        :-1
    ]

    scores, labels = object_probabilities.max(
        dim=-1
    )

    best_query = scores.argmax()

    predicted_score = scores[
        best_query
    ]

    predicted_label = labels[
        best_query
    ]

    predicted_box = pred_boxes[
        0,
        best_query
    ]

    return (
        predicted_score.cpu(),
        predicted_label.cpu(),
        predicted_box.cpu()
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

    train_dataset = SyntheticDetectionDataset(
        n_samples=600,
        image_size=64,
        seed=42
    )

    test_dataset = SyntheticDetectionDataset(
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

    image, true_label, true_box = test_dataset[0]

    print(
        "\n"
        "======================================="
    )

    print(
        "CUSTOM SIMPLE DETR"
    )

    print(
        "======================================="
    )

    model = DETR(
        in_channels=3,
        num_classes=2,
        hidden_dim=128,
        num_queries=5,
        encoder_depth=3,
        decoder_depth=3,
        num_heads=4
    ).to(device)

    print(model)

    train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        epochs=5,
        lr=1e-3
    )

    (
        predicted_score,
        predicted_label,
        predicted_box
    ) = inference(
        model=model,
        image=image,
        device=device
    )

    print(
        "\nDETR inference"
    )

    print(
        "Image shape:",
        image.shape
    )

    print(
        "True label:",
        true_label
    )

    print(
        "Predicted label:",
        predicted_label.item()
    )

    print(
        "Prediction score:",
        round(
            predicted_score.item(),
            4
        )
    )

    print(
        "True box (cx, cy, w, h):",
        true_box
    )

    print(
        "Predicted box (cx, cy, w, h):",
        predicted_box
    )

    sample_iou = box_iou(
        predicted_box.unsqueeze(
            dim=0
        ),
        true_box.unsqueeze(
            dim=0
        )
    )

    print(
        "Sample IoU:",
        round(
            sample_iou.item(),
            4
        )
    )


if __name__ == "__main__":
    main()