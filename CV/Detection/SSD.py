import random

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import ssd300_vgg16
from tqdm import tqdm

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

class ConvBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            stride: int = 1
    ):
        super().__init__()

        padding = kernel_size // 2

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(
                inplace=True
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class L2Norm(nn.Module):
    def __init__(
            self,
            channels: int,
            scale: float = 20.0
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.full(
                (channels,),
                fill_value=scale,
                dtype=torch.float32
            )
        )

    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:
        norm = torch.sqrt(
            torch.sum(
                x ** 2,
                dim=1,
                keepdim=True
            )
            + 1e-10
        )

        x = x / norm

        return (
                x
                * self.weight.view(
            1,
            -1,
            1,
            1
        )
        )

class SSDBackbone(nn.Module):
    def __init__(self):
        super().__init__()

        self.stem = nn.Sequential(
            ConvBlock(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                stride=2
            ),

            ConvBlock(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=2
            ),

            ConvBlock(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1
            )
        )

        self.norm = L2Norm(
            channels=64,
            scale=20.0
        )

        self.extra1 = ConvBlock(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            stride=2
        )

        self.extra2 = ConvBlock(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            stride=2
        )

        self.extra3 = ConvBlock(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            stride=2
        )

        self.extra4 = ConvBlock(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            stride=2
        )

        self.extra5 = ConvBlock(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            stride=2
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        feature1 = self.stem(x)
        feature1 = self.norm(feature1)

        feature2 = self.extra1(feature1)
        feature3 = self.extra2(feature2)
        feature4 = self.extra3(feature3)
        feature5 = self.extra4(feature4)
        feature6 = self.extra5(feature5)

        return [
            feature1,
            feature2,
            feature3,
            feature4,
            feature5,
            feature6
        ]

class DefaultBoxGenerator:
    def __init__(
            self,
            sizes=(16, 32, 48, 64, 96, 120, 128),
            aspect_ratios=(
                    (2.0,),
                    (2.0, 3.0),
                    (2.0, 3.0),
                    (2.0, 3.0),
                    (2.0,),
                    (2.0,)
            )
    ):

        self.sizes = sizes
        self.aspect_ratios = aspect_ratios

        self.num_defaults = [
            2 + 2 * len(ratios)
            for ratios in self.aspect_ratios
        ]

    def _boxes_for_location(
            self,
            level: int,
            center_x: float,
            center_y: float,
            image_height: int,
            image_width: int
    ):
        boxes = []

        size = float(
            self.sizes[level]
        )

        next_size = float(
            self.sizes[level + 1]
        )

        boxes.append(
            (
                center_x,
                center_y,
                size,
                size
            )
        )

        extra_size = np.sqrt(
            size
            * next_size
        )

        boxes.append(
            (
                center_x,
                center_y,
                extra_size,
                extra_size
            )
        )

        for ratio in self.aspect_ratios[level]:
            sqrt_ratio = np.sqrt(
                ratio
            )

            width = (
                    size
                    * sqrt_ratio
            )

            height = (
                    size
                    / sqrt_ratio
            )

            boxes.append(
                (
                    center_x,
                    center_y,
                    width,
                    height
                )
            )

            boxes.append(
                (
                    center_x,
                    center_y,
                    height,
                    width
                )
            )

        result = []

        for cx, cy, width, height in boxes:
            x1 = max(
                0.0,
                cx - width / 2
            )

            y1 = max(
                0.0,
                cy - height / 2
            )

            x2 = min(
                float(image_width),
                cx + width / 2
            )

            y2 = min(
                float(image_height),
                cy + height / 2
            )

            result.append(
                [x1, y1, x2, y2]
            )

        return result

    def generate(
            self,
            feature_maps,
            image_height: int,
            image_width: int
    ) -> torch.Tensor:
        all_boxes = []

        for level, feature in enumerate(feature_maps):
            _, _, height, width = feature.shape

            stride_y = (
                    image_height
                    / height
            )

            stride_x = (
                    image_width
                    / width
            )

            for y in range(height):
                center_y = (
                                   y + 0.5
                           ) * stride_y

                for x in range(width):
                    center_x = (
                                       x + 0.5
                               ) * stride_x

                    boxes = self._boxes_for_location(
                        level=level,
                        center_x=center_x,
                        center_y=center_y,
                        image_height=image_height,
                        image_width=image_width
                    )

                    all_boxes.extend(
                        boxes
                    )

        return torch.tensor(
            all_boxes,
            dtype=torch.float32,
            device=feature_maps[0].device
        )

class ClassificationHead(nn.Module):
    def __init__(
            self,
            feature_channels: int,
            num_defaults: int,
            num_classes_with_background: int
    ):
        super().__init__()

        self.num_classes = num_classes_with_background

        self.layers = nn.ModuleList()

        for channels, anchors_per_location in zip(
                feature_channels,
                num_defaults
        ):
            self.layers.append(
                nn.Conv2d(
                    in_channels=channels,
                    out_channels=(
                            anchors_per_location
                            * num_classes_with_background
                    ),
                    kernel_size=3,
                    padding=1
                )
            )

    def forward(
            self,
            features
    ) -> torch.Tensor:
        predictions = []

        for feature, layer in zip(
                features,
                self.layers
        ):
            output = layer(feature)

            batch_size = output.shape[0]

            output = output.permute(
                0,
                2,
                3,
                1
            ).contiguous()

            output = output.view(
                batch_size,
                -1,
                self.num_classes
            )

            predictions.append(
                output
            )

        return torch.cat(
            predictions,
            dim=1
        )

class LocalizationHead(nn.Module):
    def __init__(
            self,
            feature_channels,
            num_defaults
    ):
        super().__init__()

        self.layers = nn.ModuleList()

        for channels, anchors_per_location in zip(
                feature_channels,
                num_defaults
        ):
            self.layers.append(
                nn.Conv2d(
                    in_channels=channels,
                    out_channels=(
                            anchors_per_location
                            * 4
                    ),
                    kernel_size=3,
                    padding=1
                )
            )

    def forward(
            self,
            features
    ) -> torch.Tensor:
        predictions = []

        for feature, layer in zip(
                features,
                self.layers
        ):
            output = layer(feature)

            batch_size = output.shape[0]

            output = output.permute(
                0,
                2,
                3,
                1
            ).contiguous()

            output = output.view(
                batch_size,
                -1,
                4
            )

            predictions.append(
                output
            )

        return torch.cat(
            predictions,
            dim=1
        )

def box_iou(
        boxes1: torch.Tensor,
        boxes2: torch.Tensor
) -> torch.Tensor:
    area1 = (
            (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0)
            * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    )

    area2 = (
            (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0)
            * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    )

    top_left = torch.maximum(
        boxes1[:, None, :2],
        boxes2[None, :, :2]
    )

    bottom_right = torch.minimum(
        boxes1[:, None, 2:],
        boxes2[None, :, 2:]
    )

    intersection_wh = (
            bottom_right
            - top_left
    ).clamp(
        min=0
    )

    intersection = (
            intersection_wh[..., 0]
            * intersection_wh[..., 1]
    )

    union = (
            area1[:, None]
            + area2[None, :]
            - intersection
    )

    return (
            intersection
            / union.clamp(min=1e-7)
    )


def xyxy_to_cxcywh(
        boxes: torch.Tensor
) -> torch.Tensor:
    center = (
            boxes[:, :2]
            + boxes[:, 2:]
    ) / 2

    size = (
            boxes[:, 2:]
            - boxes[:, :2]
    ).clamp(
        min=1e-6
    )

    return torch.cat(
        [center, size],
        dim=1
    )


def cxcywh_to_xyxy(
        boxes: torch.Tensor
) -> torch.Tensor:
    center = boxes[:, :2]
    size = boxes[:, 2:]

    return torch.cat(
        [
            center - size / 2,
            center + size / 2
        ],
        dim=1
    )


def encode_boxes(
        ground_truth_boxes: torch.Tensor,
        default_boxes: torch.Tensor,
        center_variance: float = 0.1,
        size_variance: float = 0.2
) -> torch.Tensor:
    gt = xyxy_to_cxcywh(
        ground_truth_boxes
    )

    defaults = xyxy_to_cxcywh(
        default_boxes
    )

    encoded_center = (
            (gt[:, :2] - defaults[:, :2])
            / defaults[:, 2:]
            / center_variance
    )

    encoded_size = (
            torch.log(
                gt[:, 2:]
                / defaults[:, 2:]
            )
            / size_variance
    )

    return torch.cat(
        [
            encoded_center,
            encoded_size
        ],
        dim=1
    )


def decode_boxes(
        offsets: torch.Tensor,
        default_boxes: torch.Tensor,
        center_variance: float = 0.1,
        size_variance: float = 0.2
) -> torch.Tensor:
    defaults = xyxy_to_cxcywh(
        default_boxes
    )

    center = (
            offsets[:, :2]
            * center_variance
            * defaults[:, 2:]
            + defaults[:, :2]
    )

    size = (
            torch.exp(
                offsets[:, 2:]
                * size_variance
            )
            * defaults[:, 2:]
    )

    decoded = torch.cat(
        [center, size],
        dim=1
    )

    return cxcywh_to_xyxy(
        decoded
    )


def match_default_boxes(
        default_boxes: torch.Tensor,
        gt_boxes: torch.Tensor,
        gt_labels: torch.Tensor,
        iou_threshold: float = 0.5
):
    ious = box_iou(
        default_boxes,
        gt_boxes
    )

    best_gt_iou, best_gt_index = ious.max(
        dim=1
    )

    _, best_default_index = ious.max(
        dim=0
    )

    # Force every GT box to own at least one default box.
    best_gt_iou[
        best_default_index
    ] = 2.0

    best_gt_index[
        best_default_index
    ] = torch.arange(
        gt_boxes.shape[0],
        device=gt_boxes.device
    )

    assigned_boxes = gt_boxes[
        best_gt_index
    ]

    assigned_labels = gt_labels[
        best_gt_index
    ].clone()

    assigned_labels[
        best_gt_iou < iou_threshold
    ] = 0

    encoded_boxes = encode_boxes(
        assigned_boxes,
        default_boxes
    )

    return (
        encoded_boxes,
        assigned_labels
    )


class MultiBoxLoss(nn.Module):
    """
    SSD MultiBox loss:
      localization loss + classification loss with hard-negative mining.

    Background is class 0. Hard negatives are selected with the classic
    negative:positive ratio of 3:1.
    """

    def __init__(
            self,
            negative_positive_ratio: int = 3,
            localization_weight: float = 1.0
    ):
        super().__init__()

        self.negative_positive_ratio = negative_positive_ratio
        self.localization_weight = localization_weight

    def forward(
            self,
            class_logits: torch.Tensor,
            box_offsets: torch.Tensor,
            default_boxes: torch.Tensor,
            targets
    ):
        batch_classification_loss = 0.0
        batch_localization_loss = 0.0

        batch_size = class_logits.shape[0]

        for batch_index in range(batch_size):
            gt_boxes = targets[batch_index][
                "boxes"
            ]

            gt_labels = targets[batch_index][
                "labels"
            ]

            encoded_boxes, labels = match_default_boxes(
                default_boxes=default_boxes,
                gt_boxes=gt_boxes,
                gt_labels=gt_labels,
                iou_threshold=0.5
            )

            positive_mask = (
                    labels > 0
            )

            num_positive = int(
                positive_mask.sum().item()
            )

            num_positive_safe = max(
                num_positive,
                1
            )

            if num_positive > 0:
                localization_loss = F.smooth_l1_loss(
                    box_offsets[
                        batch_index
                    ][positive_mask],
                    encoded_boxes[
                        positive_mask
                    ],
                    reduction="sum"
                )

                localization_loss = (
                        localization_loss
                        / num_positive_safe
                )
            else:
                localization_loss = box_offsets[
                    batch_index
                ].sum() * 0.0

            per_default_classification_loss = F.cross_entropy(
                class_logits[
                    batch_index
                ],
                labels,
                reduction="none"
            )

            negative_mask = ~positive_mask

            negative_losses = per_default_classification_loss.clone()

            negative_losses[
                positive_mask
            ] = -float("inf")

            num_negative_available = int(
                negative_mask.sum().item()
            )

            num_negative = min(
                self.negative_positive_ratio
                * num_positive_safe,
                num_negative_available
            )

            hard_negative_mask = torch.zeros_like(
                negative_mask
            )

            if num_negative > 0:
                _, hard_negative_indices = torch.topk(
                    negative_losses,
                    k=num_negative
                )

                hard_negative_mask[
                    hard_negative_indices
                ] = True

            selected_mask = (
                    positive_mask
                    | hard_negative_mask
            )

            classification_loss = per_default_classification_loss[
                selected_mask
            ].sum()

            classification_loss = (
                    classification_loss
                    / num_positive_safe
            )

            batch_classification_loss += classification_loss
            batch_localization_loss += localization_loss

        classification_loss = (
                batch_classification_loss
                / batch_size
        )

        localization_loss = (
                batch_localization_loss
                / batch_size
        )

        total_loss = (
                classification_loss
                + self.localization_weight
                * localization_loss
        )

        return {
            "loss": total_loss,
            "classification_loss": classification_loss,
            "localization_loss": localization_loss
        }


def nms(
        boxes: torch.Tensor,
        scores: torch.Tensor,
        iou_threshold: float = 0.45
) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.empty(
            (0,),
            dtype=torch.long,
            device=boxes.device
        )

    order = torch.argsort(
        scores,
        descending=True
    )

    keep = []

    while order.numel() > 0:
        current = order[0]

        keep.append(
            current
        )

        if order.numel() == 1:
            break

        remaining = order[1:]

        ious = box_iou(
            boxes[
                current
            ].unsqueeze(0),
            boxes[
                remaining
            ]
        ).squeeze(0)

        order = remaining[
            ious <= iou_threshold
        ]

    return torch.stack(
        keep
    )

class SSD(nn.Module):
    def __init__(
            self,
            num_classes: int = 3
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_classes_with_background = num_classes + 1

        self.backbone = SSDBackbone()

        self.default_box_generator = DefaultBoxGenerator()

        feature_channels = (
            64,
            128,
            128,
            128,
            128,
            128
        )

        self.classification_head = ClassificationHead(
            feature_channels=feature_channels,
            num_defaults=(
                self.default_box_generator.num_defaults
            ),
            num_classes_with_background=(
                self.num_classes_with_background
            )
        )

        self.localization_head = LocalizationHead(
            feature_channels=feature_channels,
            num_defaults=(
                self.default_box_generator.num_defaults
            )
        )

        self.criterion = MultiBoxLoss(
            negative_positive_ratio=3,
            localization_weight=1.0
        )

        self._initialize_heads()

    def _initialize_heads(self):
        for module in [
            self.classification_head,
            self.localization_head
        ]:
            for layer in module.layers:
                nn.init.xavier_uniform_(
                    layer.weight
                )

                nn.init.zeros_(
                    layer.bias
                )

    def forward_raw(
            self,
            images: torch.Tensor
    ):
        features = self.backbone(
            images
        )

        class_logits = self.classification_head(
            features
        )

        box_offsets = self.localization_head(
            features
        )

        image_height = images.shape[-2]
        image_width = images.shape[-1]

        default_boxes = self.default_box_generator.generate(
            feature_maps=features,
            image_height=image_height,
            image_width=image_width
        )

        return (
            class_logits,
            box_offsets,
            default_boxes,
            features
        )

    def forward(
            self,
            images: torch.Tensor,
            targets=None
    ):
        (
            class_logits,
            box_offsets,
            default_boxes,
            _
        ) = self.forward_raw(
            images
        )

        if targets is not None:
            return self.criterion(
                class_logits=class_logits,
                box_offsets=box_offsets,
                default_boxes=default_boxes,
                targets=targets
            )

        return (
            class_logits,
            box_offsets,
            default_boxes
        )

class SyntheticDetectionDataset(Dataset):
    def __init__(
            self,
            n_samples: int = 400,
            image_size: int = 128,
            num_classes: int = 3,
            seed: int = 42
    ):
        super().__init__()

        self.n_samples = n_samples
        self.image_size = image_size
        self.num_classes = num_classes

        generator = torch.Generator()
        generator.manual_seed(
            seed
        )

        images = []
        targets = []

        class_colors = torch.tensor(
            [
                [1.0, 0.2, 0.2],
                [0.2, 1.0, 0.2],
                [0.2, 0.2, 1.0]
            ],
            dtype=torch.float32
        )

        for _ in range(n_samples):
            image = (
                    torch.rand(
                        3,
                        image_size,
                        image_size,
                        generator=generator
                    )
                    * 0.08
            )

            class_index = int(
                torch.randint(
                    low=0,
                    high=num_classes,
                    size=(1,),
                    generator=generator
                ).item()
            )

            box_height = int(
                torch.randint(
                    low=image_size // 8,
                    high=image_size // 2,
                    size=(1,),
                    generator=generator
                ).item()
            )

            box_width = int(
                torch.randint(
                    low=image_size // 8,
                    high=image_size // 2,
                    size=(1,),
                    generator=generator
                ).item()
            )

            y1 = int(
                torch.randint(
                    low=0,
                    high=(
                        image_size
                        - box_height
                        + 1
                    ),
                    size=(1,),
                    generator=generator
                ).item()
            )

            x1 = int(
                torch.randint(
                    low=0,
                    high=(
                        image_size
                        - box_width
                        + 1
                    ),
                    size=(1,),
                    generator=generator
                ).item()
            )

            x2 = x1 + box_width
            y2 = y1 + box_height

            color = class_colors[
                class_index
            ].view(
                3,
                1,
                1
            )

            image[
                :,
                y1:y2,
                x1:x2
            ] = (
                    color
                    + torch.rand(
                        3,
                        box_height,
                        box_width,
                        generator=generator
                    )
                    * 0.05
            ).clamp(
                0,
                1
            )

            images.append(
                image
            )

            targets.append(
                {
                    "boxes": torch.tensor(
                        [
                            [
                                float(x1),
                                float(y1),
                                float(x2),
                                float(y2)
                            ]
                        ],
                        dtype=torch.float32
                    ),

                    # 0 is reserved for background in SSD.
                    "labels": torch.tensor(
                        [class_index + 1],
                        dtype=torch.long
                    )
                }
            )

        self.images = torch.stack(
            images
        )

        self.targets = targets

    def __len__(self):
        return self.n_samples

    def __getitem__(
            self,
            index
    ):
        target = {
            "boxes": self.targets[index][
                "boxes"
            ].clone(),

            "labels": self.targets[index][
                "labels"
            ].clone()
        }

        return (
            self.images[index],
            target
        )


def detection_collate_fn(batch):
    images = torch.stack(
        [
            item[0]
            for item in batch
        ]
    )

    targets = [
        item[1]
        for item in batch
    ]

    return images, targets


def move_targets_to_device(
        targets,
        device
):
    return [
        {
            "boxes": target[
                "boxes"
            ].to(device),

            "labels": target[
                "labels"
            ].to(device)
        }
        for target in targets
    ]


@torch.no_grad()
def ssd_inference(
        model: SSD,
        images: torch.Tensor,
        device,
        score_threshold: float = 0.25,
        nms_threshold: float = 0.45,
        top_k: int = 100
):
    model.eval()

    images = images.to(
        device
    )

    (
        class_logits,
        box_offsets,
        default_boxes
    ) = model(
        images
    )

    probabilities = torch.softmax(
        class_logits,
        dim=-1
    )

    results = []

    image_height = images.shape[-2]
    image_width = images.shape[-1]

    for batch_index in range(
            images.shape[0]
    ):
        decoded_boxes = decode_boxes(
            offsets=box_offsets[
                batch_index
            ],
            default_boxes=default_boxes
        )

        decoded_boxes[:, 0::2] = decoded_boxes[:, 0::2].clamp(
            0,
            image_width
        )

        decoded_boxes[:, 1::2] = decoded_boxes[:, 1::2].clamp(
            0,
            image_height
        )

        final_boxes = []
        final_scores = []
        final_labels = []

        # class 0 is background
        for class_index in range(
                1,
                model.num_classes_with_background
        ):
            scores = probabilities[
                batch_index,
                :,
                class_index
            ]

            mask = (
                    scores
                    >= score_threshold
            )

            if not mask.any():
                continue

            class_boxes = decoded_boxes[
                mask
            ]

            class_scores = scores[
                mask
            ]

            keep = nms(
                boxes=class_boxes,
                scores=class_scores,
                iou_threshold=nms_threshold
            )

            final_boxes.append(
                class_boxes[
                    keep
                ]
            )

            final_scores.append(
                class_scores[
                    keep
                ]
            )

            final_labels.append(
                torch.full(
                    (keep.numel(),),
                    fill_value=class_index,
                    dtype=torch.long,
                    device=device
                )
            )

        if len(final_boxes) == 0:
            results.append(
                {
                    "boxes": torch.empty(
                        (0, 4),
                        device=device
                    ),
                    "scores": torch.empty(
                        (0,),
                        device=device
                    ),
                    "labels": torch.empty(
                        (0,),
                        dtype=torch.long,
                        device=device
                    )
                }
            )

            continue

        final_boxes = torch.cat(
            final_boxes,
            dim=0
        )

        final_scores = torch.cat(
            final_scores,
            dim=0
        )

        final_labels = torch.cat(
            final_labels,
            dim=0
        )

        order = torch.argsort(
            final_scores,
            descending=True
        )[:top_k]

        results.append(
            {
                "boxes": final_boxes[
                    order
                ],
                "scores": final_scores[
                    order
                ],
                "labels": final_labels[
                    order
                ]
            }
        )

    return results


def best_detection_metrics(
        prediction,
        target
):
    if prediction[
        "boxes"
    ].shape[0] == 0:
        return 0.0, 0.0, 0.0

    best_index = torch.argmax(
        prediction[
            "scores"
        ]
    )

    predicted_box = prediction[
        "boxes"
    ][best_index].unsqueeze(0)

    predicted_label = prediction[
        "labels"
    ][best_index]

    target_box = target[
        "boxes"
    ][:1]

    target_label = target[
        "labels"
    ][0]

    iou = box_iou(
        predicted_box,
        target_box
    )[0, 0].item()

    class_correct = float(
        predicted_label.item()
        == target_label.item()
    )

    detection_correct = float(
        iou >= 0.5
        and class_correct == 1.0
    )

    return (
        iou,
        class_correct,
        detection_correct
    )


def train_epoch(
        model: SSD,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device
):
    model.train()

    total_loss = 0.0
    total_classification_loss = 0.0
    total_localization_loss = 0.0
    total = 0

    for images, targets in tqdm(
            dataloader,
            desc="Training custom SSD"
    ):
        images = images.to(
            device
        )

        targets = move_targets_to_device(
            targets,
            device
        )

        optimizer.zero_grad()

        loss_dict = model(
            images,
            targets=targets
        )

        loss = loss_dict[
            "loss"
        ]

        loss.backward()

        optimizer.step()

        batch_size = images.shape[0]

        total_loss += (
                loss.item()
                * batch_size
        )

        total_classification_loss += (
                loss_dict[
                    "classification_loss"
                ].item()
                * batch_size
        )

        total_localization_loss += (
                loss_dict[
                    "localization_loss"
                ].item()
                * batch_size
        )

        total += batch_size

    return (
        total_loss / total,
        total_classification_loss / total,
        total_localization_loss / total
    )


@torch.no_grad()
def evaluate_custom(
        model: SSD,
        dataloader: DataLoader,
        device
):
    model.eval()

    total_loss = 0.0
    total_iou = 0.0
    total_class_accuracy = 0.0
    total_detection_accuracy = 0.0
    total = 0

    for images, targets in tqdm(
            dataloader,
            desc="Validation custom SSD"
    ):
        images = images.to(
            device
        )

        device_targets = move_targets_to_device(
            targets,
            device
        )

        loss_dict = model(
            images,
            targets=device_targets
        )

        predictions = ssd_inference(
            model=model,
            images=images,
            device=device,
            score_threshold=0.10
        )

        for prediction, target in zip(
                predictions,
                device_targets
        ):
            iou, class_correct, detection_correct = best_detection_metrics(
                prediction,
                target
            )

            total_iou += iou
            total_class_accuracy += class_correct
            total_detection_accuracy += detection_correct
            total += 1

        total_loss += (
                loss_dict[
                    "loss"
                ].item()
                * images.shape[0]
        )

    return (
        total_loss / total,
        total_iou / total,
        total_class_accuracy / total,
        total_detection_accuracy / total
    )


def train_custom_model(
        model: SSD,
        train_loader: DataLoader,
        test_loader: DataLoader,
        device,
        epochs: int = 3,
        lr: float = 1e-3
):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr
    )

    final_metrics = None

    for epoch in range(epochs):
        (
            train_loss,
            train_cls,
            train_loc
        ) = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device
        )

        (
            test_loss,
            test_iou,
            test_class_accuracy,
            test_detection_accuracy
        ) = evaluate_custom(
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
            f"Train classification loss: {train_cls:.4f}"
        )

        print(
            f"Train localization loss: {train_loc:.4f}"
        )

        print(
            f"Test loss: {test_loss:.4f}"
        )

        print(
            f"Test mean IoU: {test_iou:.4f}"
        )

        print(
            f"Test class accuracy: {test_class_accuracy:.4f}"
        )

        print(
            f"Test detection accuracy@IoU0.5: "
            f"{test_detection_accuracy:.4f}"
        )

        print()

        final_metrics = (
            test_iou,
            test_class_accuracy,
            test_detection_accuracy
        )

    return final_metrics


def train_torchvision_epoch(
        model,
        dataloader,
        optimizer,
        device
):
    model.train()

    total_loss = 0.0
    total = 0

    for images, targets in tqdm(
            dataloader,
            desc="Training torchvision SSD300"
    ):
        image_list = [
            image.to(device)
            for image in images
        ]

        device_targets = move_targets_to_device(
            targets,
            device
        )

        optimizer.zero_grad()

        loss_dict = model(
            image_list,
            device_targets
        )

        loss = sum(
            loss_dict.values()
        )

        loss.backward()
        optimizer.step()

        batch_size = len(
            image_list
        )

        total_loss += (
                loss.item()
                * batch_size
        )

        total += batch_size

    return total_loss / total


@torch.no_grad()
def evaluate_torchvision(
        model,
        dataloader,
        device,
        score_threshold: float = 0.10
):
    model.eval()

    total_iou = 0.0
    total_class_accuracy = 0.0
    total_detection_accuracy = 0.0
    total = 0

    for images, targets in tqdm(
            dataloader,
            desc="Validation torchvision SSD300"
    ):
        image_list = [
            image.to(device)
            for image in images
        ]

        predictions = model(
            image_list
        )

        device_targets = move_targets_to_device(
            targets,
            device
        )

        for prediction, target in zip(
                predictions,
                device_targets
        ):
            mask = (
                    prediction[
                        "scores"
                    ]
                    >= score_threshold
            )

            filtered_prediction = {
                "boxes": prediction[
                    "boxes"
                ][mask],
                "scores": prediction[
                    "scores"
                ][mask],
                "labels": prediction[
                    "labels"
                ][mask]
            }

            iou, class_correct, detection_correct = best_detection_metrics(
                filtered_prediction,
                target
            )

            total_iou += iou
            total_class_accuracy += class_correct
            total_detection_accuracy += detection_correct
            total += 1

    return (
        total_iou / total,
        total_class_accuracy / total,
        total_detection_accuracy / total
    )


def train_torchvision_model(
        model,
        train_loader,
        test_loader,
        device,
        epochs: int = 3,
        lr: float = 1e-3
):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr
    )

    final_metrics = None

    for epoch in range(epochs):
        train_loss = train_torchvision_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device
        )

        (
            test_iou,
            test_class_accuracy,
            test_detection_accuracy
        ) = evaluate_torchvision(
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
            f"Test mean IoU: {test_iou:.4f}"
        )

        print(
            f"Test class accuracy: {test_class_accuracy:.4f}"
        )

        print(
            f"Test detection accuracy@IoU0.5: "
            f"{test_detection_accuracy:.4f}"
        )

        print()

        final_metrics = (
            test_iou,
            test_class_accuracy,
            test_detection_accuracy
        )

    return final_metrics


def print_feature_shapes(
        model: SSD,
        image: torch.Tensor,
        device
):
    model.eval()

    with torch.no_grad():
        features = model.backbone(
            image.unsqueeze(0).to(device)
        )

    print(
        "Feature map shapes:"
    )

    for index, feature in enumerate(
            features,
            start=1
    ):
        print(
            f"F{index}: {tuple(feature.shape)}"
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
        n_samples=300,
        image_size=128,
        num_classes=3,
        seed=42
    )

    test_dataset = SyntheticDetectionDataset(
        n_samples=100,
        image_size=128,
        num_classes=3,
        seed=1337
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=detection_collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=detection_collate_fn
    )

    image, target = test_dataset[0]

    print(
        "\n"
        "======================================="
    )

    print(
        "CUSTOM EDUCATIONAL SSD"
    )

    print(
        "======================================="
    )

    custom_ssd = SSD(
        num_classes=3
    ).to(
        device
    )

    print(
        custom_ssd
    )

    print_feature_shapes(
        model=custom_ssd,
        image=image,
        device=device
    )

    with torch.no_grad():
        (
            class_logits,
            box_offsets,
            default_boxes,
            _
        ) = custom_ssd.forward_raw(
            image.unsqueeze(0).to(device)
        )

    print(
        "\nRaw SSD outputs:"
    )

    print(
        "Class logits shape:",
        class_logits.shape
    )

    print(
        "Box offsets shape:",
        box_offsets.shape
    )

    print(
        "Default boxes shape:",
        default_boxes.shape
    )

    custom_metrics = train_custom_model(
        model=custom_ssd,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        epochs=3,
        lr=1e-3
    )

    predictions = ssd_inference(
        model=custom_ssd,
        images=image.unsqueeze(0),
        device=device,
        score_threshold=0.10
    )

    prediction = predictions[0]

    print(
        "\nCustom SSD inference"
    )

    print(
        "True box:",
        target[
            "boxes"
        ]
    )

    print(
        "True label:",
        target[
            "labels"
        ]
    )

    print(
        "Predicted boxes:",
        prediction[
            "boxes"
        ][:5].cpu()
    )

    print(
        "Predicted labels:",
        prediction[
            "labels"
        ][:5].cpu()
    )

    print(
        "Predicted scores:",
        prediction[
            "scores"
        ][:5].cpu()
    )

    print(
        "\n"
        "======================================="
    )

    print(
        "TORCHVISION SSD300 VGG16"
    )

    print(
        "======================================="
    )

    if ssd300_vgg16 is None:
        print(
            "torchvision SSD is unavailable in this environment."
        )

        return

    framework_ssd = ssd300_vgg16(
        weights=None,
        weights_backbone=None,
        num_classes=4
    ).to(
        device
    )

    print(
        framework_ssd
    )

    framework_metrics = train_torchvision_model(
        model=framework_ssd,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        epochs=3,
        lr=1e-3
    )

    framework_ssd.eval()

    with torch.no_grad():
        framework_prediction = framework_ssd(
            [
                image.to(device)
            ]
        )[0]

    print(
        "\nTorchvision SSD300 inference"
    )

    print(
        "Predicted boxes:",
        framework_prediction[
            "boxes"
        ][:5].cpu()
    )

    print(
        "Predicted labels:",
        framework_prediction[
            "labels"
        ][:5].cpu()
    )

    print(
        "Predicted scores:",
        framework_prediction[
            "scores"
        ][:5].cpu()
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
        f"Custom SSD mean IoU:         "
        f"{custom_metrics[0]:.4f}"
    )

    print(
        f"Torchvision SSD300 mean IoU: "
        f"{framework_metrics[0]:.4f}"
    )

    print()

    print(
        f"Custom SSD class accuracy:         "
        f"{custom_metrics[1]:.4f}"
    )

    print(
        f"Torchvision SSD300 class accuracy: "
        f"{framework_metrics[1]:.4f}"
    )

    print()

    print(
        f"Custom SSD detection accuracy@0.5:         "
        f"{custom_metrics[2]:.4f}"
    )

    print(
        f"Torchvision SSD300 detection accuracy@0.5: "
        f"{framework_metrics[2]:.4f}"
    )


if __name__ == "__main__":
    main()
