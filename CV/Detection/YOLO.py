import copy
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
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
            stride: int = 1,
            groups: int = 1
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
                groups=groups,
                bias=False
            ),

            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Bottleneck(nn.Module):
    def __init__(
            self,
            channels: int,
            shortcut: bool = True
    ):
        super().__init__()

        self.conv1 = ConvBlock(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3
        )

        self.conv2 = ConvBlock(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3
        )

        self.shortcut = shortcut

    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:
        y = self.conv1(x)
        y = self.conv2(y)

        if self.shortcut:
            y = y + x

        return y


class C3k2(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            n: int = 2,
            expansion: float = 0.5
    ):
        super().__init__()

        hidden_channels = int(out_channels * expansion)

        self.conv1 = ConvBlock(
            in_channels=in_channels,
            out_channels=hidden_channels * 2,
            kernel_size=1
        )

        self.blocks = nn.ModuleList([
            Bottleneck(
                channels=hidden_channels,
                shortcut=True
            )
            for _ in range(n)
        ])

        self.conv2 = ConvBlock(
            in_channels=hidden_channels * (2 + n),
            out_channels=out_channels,
            kernel_size=1
        )

    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:
        x = self.conv1(x)

        left, right = x.chunk(chunks=2, dim=1)

        features = [left, right]

        for block in self.blocks:
            right = block(right)
            features.append(right)

        x = torch.cat(features, dim=1)

        return self.conv2(x)


class SPPF(nn.Module):
    """
    Spatial Pyramid Pooling Fast.
    """

    def __init__(
            self,
            channels: int
    ):
        super().__init__()

        hidden_channels = channels // 2

        self.conv1 = ConvBlock(
            in_channels=channels,
            out_channels=hidden_channels,
            kernel_size=1
        )

        self.pool = nn.MaxPool2d(
            kernel_size=5,
            stride=1,
            padding=2
        )

        self.conv2 = ConvBlock(
            in_channels=hidden_channels * 4,
            out_channels=channels,
            kernel_size=1
        )

    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:
        identity = x

        x = self.conv1(x)

        p1 = self.pool(x)
        p2 = self.pool(p1)
        p3 = self.pool(p2)

        x = torch.cat(
            (x, p1, p2, p3),
            dim=1
        )

        x = self.conv2(x)

        return x + identity


class PSABlock(nn.Module):
    """
    Position-Sensitive Attention block.

    Attention работает по пространственным токенам H*W,
    после чего используется convolutional FFN.
    """

    def __init__(
            self,
            channels: int,
            num_heads: int = 4
    ):
        super().__init__()

        if channels % num_heads != 0:
            raise ValueError(
                "channels must be divisible by num_heads"
            )

        self.attention = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True
        )

        self.ffn = nn.Sequential(
            ConvBlock(
                in_channels=channels,
                out_channels=channels * 2,
                kernel_size=1
            ),

            nn.Conv2d(
                in_channels=channels * 2,
                out_channels=channels,
                kernel_size=1
            )
        )

    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:
        batch_size, channels, height, width = x.shape

        tokens = x.flatten(
            start_dim=2
        ).transpose(
            1,
            2
        )

        attention_output, _ = self.attention(
            tokens,
            tokens,
            tokens,
            need_weights=False
        )

        tokens = tokens + attention_output

        x = tokens.transpose(
            1,
            2
        ).reshape(
            batch_size,
            channels,
            height,
            width
        )

        return x + self.ffn(x)


class C2PSA(nn.Module):
    def __init__(
            self,
            channels: int,
            n: int = 1
    ):
        super().__init__()

        hidden_channels = channels // 2

        self.conv1 = ConvBlock(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1
        )

        self.blocks = nn.Sequential(*[
            PSABlock(
                channels=hidden_channels,
                num_heads=4
            )
            for _ in range(n)
        ])

        self.conv2 = ConvBlock(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1
        )

    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:
        x = self.conv1(x)

        left, right = x.chunk(
            chunks=2,
            dim=1
        )

        right = self.blocks(right)

        x = torch.cat(
            (left, right),
            dim=1
        )

        return self.conv2(x)


class YOLO26Backbone(nn.Module):
    def __init__(
            self,
            base_channels: int = 16
    ):
        super().__init__()

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8
        c5 = base_channels * 16

        self.stem1 = ConvBlock(
            in_channels=3,
            out_channels=c1,
            kernel_size=3,
            stride=2
        )

        self.stem2 = ConvBlock(
            in_channels=c1,
            out_channels=c2,
            kernel_size=3,
            stride=2
        )

        self.stage2 = C3k2(
            in_channels=c2,
            out_channels=c3,
            n=1
        )

        self.down3 = ConvBlock(
            in_channels=c3,
            out_channels=c3,
            kernel_size=3,
            stride=2
        )

        self.stage3 = C3k2(
            in_channels=c3,
            out_channels=c3,
            n=2
        )

        self.down4 = ConvBlock(
            in_channels=c3,
            out_channels=c4,
            kernel_size=3,
            stride=2
        )

        self.stage4 = C3k2(
            in_channels=c4,
            out_channels=c4,
            n=2
        )

        self.down5 = ConvBlock(
            in_channels=c4,
            out_channels=c5,
            kernel_size=3,
            stride=2
        )

        self.stage5 = C3k2(
            in_channels=c5,
            out_channels=c5,
            n=2
        )

        self.sppf = SPPF(
            channels=c5
        )

        self.c2psa = C2PSA(
            channels=c5,
            n=1
        )

        self.out_channels = (
            c3,
            c4,
            c5
        )

    def forward(
            self,
            x: torch.Tensor
    ):
        x = self.stem1(x)
        x = self.stem2(x)
        x = self.stage2(x)

        x = self.down3(x)
        p3 = self.stage3(x)

        x = self.down4(p3)
        p4 = self.stage4(x)

        x = self.down5(p4)
        x = self.stage5(x)
        x = self.sppf(x)
        p5 = self.c2psa(x)

        return (
            p3,
            p4,
            p5
        )


class YOLO26Neck(nn.Module):
    """
    FPN top-down + PAN bottom-up.
    """

    def __init__(
            self,
            channels
    ):
        super().__init__()

        c3, c4, c5 = channels

        self.up = nn.Upsample(
            scale_factor=2,
            mode="nearest"
        )

        self.fpn4 = C3k2(
            in_channels=c5 + c4,
            out_channels=c4,
            n=2
        )

        self.fpn3 = C3k2(
            in_channels=c4 + c3,
            out_channels=c3,
            n=2
        )

        self.down4 = ConvBlock(
            in_channels=c3,
            out_channels=c4,
            kernel_size=3,
            stride=2
        )

        self.pan4 = C3k2(
            in_channels=c4 + c4,
            out_channels=c4,
            n=2
        )

        self.down5 = ConvBlock(
            in_channels=c4,
            out_channels=c5,
            kernel_size=3,
            stride=2
        )

        self.pan5 = C3k2(
            in_channels=c5 + c5,
            out_channels=c5,
            n=2
        )

        self.out_channels = (
            c3,
            c4,
            c5
        )

    def forward(
            self,
            features
    ):
        p3, p4, p5 = features

        n4 = self.up(p5)
        n4 = torch.cat(
            (n4, p4),
            dim=1
        )
        n4 = self.fpn4(n4)

        n3 = self.up(n4)
        n3 = torch.cat(
            (n3, p3),
            dim=1
        )
        n3 = self.fpn3(n3)

        p4_out = self.down4(n3)
        p4_out = torch.cat(
            (p4_out, n4),
            dim=1
        )
        p4_out = self.pan4(p4_out)

        p5_out = self.down5(p4_out)
        p5_out = torch.cat(
            (p5_out, p5),
            dim=1
        )
        p5_out = self.pan5(p5_out)

        return (
            n3,
            p4_out,
            p5_out
        )


class DWConvBlock(nn.Module):
    def __init__(
            self,
            channels: int
    ):
        super().__init__()

        self.block = nn.Sequential(
            ConvBlock(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                groups=channels
            ),

            ConvBlock(
                in_channels=channels,
                out_channels=channels,
                kernel_size=1
            )
        )

    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:
        return self.block(x)


class DetectionScaleHead(nn.Module):
    """
    Decoupled head:
        box branch -> 4 direct box values
        class branch -> num_classes logits

    В YOLO26 reg_max=1, поэтому DFL отсутствует.
    """

    def __init__(
            self,
            channels: int,
            num_classes: int
    ):
        super().__init__()

        self.box_head = nn.Sequential(
            ConvBlock(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3
            ),

            ConvBlock(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3
            ),

            nn.Conv2d(
                in_channels=channels,
                out_channels=4,
                kernel_size=1
            )
        )

        self.class_head = nn.Sequential(
            DWConvBlock(
                channels=channels
            ),

            DWConvBlock(
                channels=channels
            ),

            nn.Conv2d(
                in_channels=channels,
                out_channels=num_classes,
                kernel_size=1
            )
        )

    def forward(
            self,
            x: torch.Tensor
    ):
        return {
            "box": self.box_head(x),
            "cls": self.class_head(x)
        }


class Detect26Branch(nn.Module):
    def __init__(
            self,
            channels,
            num_classes: int
    ):
        super().__init__()

        self.heads = nn.ModuleList([
            DetectionScaleHead(
                channels=channel,
                num_classes=num_classes
            )
            for channel in channels
        ])

    def forward(
            self,
            features
    ):
        return [
            head(feature)
            for head, feature in zip(
                self.heads,
                features
            )
        ]


class Detect26(nn.Module):
    """
    Учебная dual-head схема YOLO26.

    one_to_many:
        много кандидатов на объект; production YOLO использует NMS.

    one_to_one:
        end-to-end ветка; целится в одну детекцию на объект и может
        использоваться без NMS.
    """

    def __init__(
            self,
            channels,
            num_classes: int
    ):
        super().__init__()

        self.one_to_many = Detect26Branch(
            channels=channels,
            num_classes=num_classes
        )

        self.one_to_one = copy.deepcopy(
            self.one_to_many
        )

    def forward(
            self,
            features
    ):
        return {
            "one_to_many": self.one_to_many(features),
            "one_to_one": self.one_to_one(features)
        }


class YOLO26(nn.Module):
    def __init__(
            self,
            num_classes: int = 3,
            base_channels: int = 16
    ):
        super().__init__()

        self.num_classes = num_classes

        self.backbone = YOLO26Backbone(
            base_channels=base_channels
        )

        self.neck = YOLO26Neck(
            channels=self.backbone.out_channels
        )

        self.detect = Detect26(
            channels=self.neck.out_channels,
            num_classes=num_classes
        )

    def forward(
            self,
            x: torch.Tensor
    ):
        features = self.backbone(x)
        features = self.neck(features)
        return self.detect(features)


class SyntheticDetectionDataset(Dataset):
    def __init__(
            self,
            n_samples: int = 600,
            image_size: int = 128,
            num_classes: int = 3,
            seed: int = 42
    ):
        super().__init__()

        self.n_samples = n_samples
        self.image_size = image_size
        self.num_classes = num_classes

        generator = torch.Generator()
        generator.manual_seed(seed)

        images = []
        boxes = []
        labels = []

        for _ in range(n_samples):
            image = torch.randn(
                3,
                image_size,
                image_size,
                generator=generator
            ) * 0.05

            image = image.clamp(
                0,
                1
            )

            label = int(
                torch.randint(
                    low=0,
                    high=num_classes,
                    size=(1,),
                    generator=generator
                ).item()
            )

            rect_height = int(
                torch.randint(
                    low=image_size // 7,
                    high=image_size // 2,
                    size=(1,),
                    generator=generator
                ).item()
            )

            rect_width = int(
                torch.randint(
                    low=image_size // 7,
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

            image[
                label,
                y1:y2,
                x1:x2
            ] += 0.95

            for channel in range(3):
                if channel != label:
                    image[
                        channel,
                        y1:y2,
                        x1:x2
                    ] += 0.15

            image = image.clamp(
                0,
                1
            )

            box = torch.tensor([
                x1 / image_size,
                y1 / image_size,
                x2 / image_size,
                y2 / image_size
            ], dtype=torch.float32)

            images.append(image)
            boxes.append(box)
            labels.append(label)

        self.images = torch.stack(images)
        self.boxes = torch.stack(boxes)
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
            self.boxes[index],
            self.labels[index]
        )


def decode_scale_boxes(
        raw_boxes: torch.Tensor
) -> torch.Tensor:
    """
    raw_boxes:
        (B, 4, H, W)

    Голова предсказывает положительные расстояния
    left/top/right/bottom от центра grid-cell в единицах клеток.

    Это упрощённая DFL-free anchor-free регрессия.
    """

    batch_size, _, height, width = raw_boxes.shape

    device = raw_boxes.device
    dtype = raw_boxes.dtype

    grid_y, grid_x = torch.meshgrid(
        torch.arange(
            height,
            device=device,
            dtype=dtype
        ),
        torch.arange(
            width,
            device=device,
            dtype=dtype
        ),
        indexing="ij"
    )

    center_x = (
            grid_x
            + 0.5
    ) / width

    center_y = (
            grid_y
            + 0.5
    ) / height

    distances = F.softplus(
        raw_boxes
    )

    left = distances[:, 0] / width
    top = distances[:, 1] / height
    right = distances[:, 2] / width
    bottom = distances[:, 3] / height

    x1 = center_x.unsqueeze(0) - left
    y1 = center_y.unsqueeze(0) - top
    x2 = center_x.unsqueeze(0) + right
    y2 = center_y.unsqueeze(0) + bottom

    boxes = torch.stack(
        (x1, y1, x2, y2),
        dim=1
    )

    return boxes.clamp(
        0,
        1
    )


def aligned_iou(
        boxes1: torch.Tensor,
        boxes2: torch.Tensor,
        eps: float = 1e-7
) -> torch.Tensor:
    """
    boxes1, boxes2:
        (N, 4), xyxy

    Возвращает IoU для соответствующих пар boxes1[i], boxes2[i].
    """

    x1 = torch.maximum(
        boxes1[:, 0],
        boxes2[:, 0]
    )

    y1 = torch.maximum(
        boxes1[:, 1],
        boxes2[:, 1]
    )

    x2 = torch.minimum(
        boxes1[:, 2],
        boxes2[:, 2]
    )

    y2 = torch.minimum(
        boxes1[:, 3],
        boxes2[:, 3]
    )

    intersection = (
            (x2 - x1).clamp(min=0)
            * (y2 - y1).clamp(min=0)
    )

    area1 = (
            (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0)
            * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    )

    area2 = (
            (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0)
            * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    )

    union = (
            area1
            + area2
            - intersection
    )

    return (
            intersection
            / (union + eps)
    )


def select_scale(
        box: torch.Tensor
) -> int:
    width = (
            box[2]
            - box[0]
    ).item()

    height = (
            box[3]
            - box[1]
    ).item()

    size = max(
        width,
        height
    )

    if size < 0.25:
        return 0

    if size < 0.40:
        return 1

    return 2


def focal_binary_loss(
        logits: torch.Tensor,
        targets: torch.Tensor,
        alpha: float = 0.75,
        gamma: float = 2.0
):
    bce = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none"
    )

    probabilities = torch.sigmoid(
        logits
    )

    p_t = (
            probabilities * targets
            + (1.0 - probabilities) * (1.0 - targets)
    )

    alpha_t = (
            alpha * targets
            + (1.0 - alpha) * (1.0 - targets)
    )

    loss = (
            alpha_t
            * (1.0 - p_t).pow(gamma)
            * bce
    )

    return loss.mean()


def branch_loss(
        predictions,
        boxes: torch.Tensor,
        labels: torch.Tensor,
        num_classes: int,
        one_to_many: bool
):
    device = boxes.device

    cls_targets = []
    positive_masks = []
    box_targets = []

    for prediction in predictions:
        batch_size, _, height, width = prediction["cls"].shape

        cls_target = torch.zeros(
            batch_size,
            num_classes,
            height,
            width,
            device=device
        )

        positive_mask = torch.zeros(
            batch_size,
            height,
            width,
            dtype=torch.bool,
            device=device
        )

        box_target = torch.zeros(
            batch_size,
            4,
            height,
            width,
            device=device
        )

        cls_targets.append(cls_target)
        positive_masks.append(positive_mask)
        box_targets.append(box_target)

    for batch_index in range(boxes.size(0)):
        box = boxes[batch_index]
        label = int(
            labels[batch_index].item()
        )

        scale_index = select_scale(box)

        _, _, height, width = predictions[
            scale_index
        ]["cls"].shape

        center_x = (
                box[0]
                + box[2]
        ) / 2.0

        center_y = (
                box[1]
                + box[3]
        ) / 2.0

        grid_x = min(
            int(center_x.item() * width),
            width - 1
        )

        grid_y = min(
            int(center_y.item() * height),
            height - 1
        )

        if one_to_many:
            offsets = [
                (-1, 0),
                (0, -1),
                (0, 0),
                (0, 1),
                (1, 0)
            ]
        else:
            offsets = [
                (0, 0)
            ]

        for dy, dx in offsets:
            y = grid_y + dy
            x = grid_x + dx

            if not (
                    0 <= y < height
                    and 0 <= x < width
            ):
                continue

            cls_targets[
                scale_index
            ][
                batch_index,
                label,
                y,
                x
            ] = 1.0

            positive_masks[
                scale_index
            ][
                batch_index,
                y,
                x
            ] = True

            box_targets[
                scale_index
            ][
                batch_index,
                :,
                y,
                x
            ] = box

    cls_loss = 0.0
    box_l1_loss = 0.0
    box_iou_loss = 0.0
    positive_count = 0

    for prediction, cls_target, positive_mask, box_target in zip(
            predictions,
            cls_targets,
            positive_masks,
            box_targets
    ):
        cls_loss = (
                cls_loss
                + focal_binary_loss(
                    prediction["cls"],
                    cls_target
                )
        )

        if positive_mask.any():
            decoded_boxes = decode_scale_boxes(
                prediction["box"]
            )

            predicted_positive_boxes = decoded_boxes.permute(
                0,
                2,
                3,
                1
            )[positive_mask]

            target_positive_boxes = box_target.permute(
                0,
                2,
                3,
                1
            )[positive_mask]

            count = predicted_positive_boxes.size(0)

            box_l1_loss = (
                    box_l1_loss
                    + F.smooth_l1_loss(
                        predicted_positive_boxes,
                        target_positive_boxes,
                        reduction="sum"
                    )
            )

            iou = aligned_iou(
                predicted_positive_boxes,
                target_positive_boxes
            )

            box_iou_loss = (
                    box_iou_loss
                    + (1.0 - iou).sum()
            )

            positive_count += count

    if positive_count > 0:
        box_l1_loss = (
                box_l1_loss
                / positive_count
        )

        box_iou_loss = (
                box_iou_loss
                / positive_count
        )

    total_loss = (
            cls_loss
            + 2.0 * box_l1_loss
            + 2.0 * box_iou_loss
    )

    return (
        total_loss,
        cls_loss,
        box_l1_loss,
        box_iou_loss
    )


def yolo26_loss(
        outputs,
        boxes: torch.Tensor,
        labels: torch.Tensor,
        num_classes: int,
        one_to_one_weight: float = 1.0
):
    one_to_many_loss, _, _, _ = branch_loss(
        predictions=outputs["one_to_many"],
        boxes=boxes,
        labels=labels,
        num_classes=num_classes,
        one_to_many=True
    )

    one_to_one_loss, _, _, _ = branch_loss(
        predictions=outputs["one_to_one"],
        boxes=boxes,
        labels=labels,
        num_classes=num_classes,
        one_to_many=False
    )

    total_loss = (
            one_to_many_loss
            + one_to_one_weight * one_to_one_loss
    )

    return (
        total_loss,
        one_to_many_loss,
        one_to_one_loss
    )


@torch.no_grad()
def decode_one_to_one(
        outputs
):
    predictions = outputs["one_to_one"]

    all_boxes = []
    all_scores = []
    all_classes = []

    for prediction in predictions:
        boxes = decode_scale_boxes(
            prediction["box"]
        )

        class_probabilities = torch.sigmoid(
            prediction["cls"]
        )

        best_scores, best_classes = class_probabilities.max(
            dim=1
        )

        batch_size = boxes.size(0)

        boxes = boxes.permute(
            0,
            2,
            3,
            1
        ).reshape(
            batch_size,
            -1,
            4
        )

        best_scores = best_scores.reshape(
            batch_size,
            -1
        )

        best_classes = best_classes.reshape(
            batch_size,
            -1
        )

        all_boxes.append(boxes)
        all_scores.append(best_scores)
        all_classes.append(best_classes)

    all_boxes = torch.cat(
        all_boxes,
        dim=1
    )

    all_scores = torch.cat(
        all_scores,
        dim=1
    )

    all_classes = torch.cat(
        all_classes,
        dim=1
    )

    best_indices = all_scores.argmax(
        dim=1
    )

    batch_indices = torch.arange(
        all_scores.size(0),
        device=all_scores.device
    )

    boxes = all_boxes[
        batch_indices,
        best_indices
    ]

    scores = all_scores[
        batch_indices,
        best_indices
    ]

    classes = all_classes[
        batch_indices,
        best_indices
    ]

    return (
        boxes,
        scores,
        classes
    )


@torch.no_grad()
def detection_metrics(
        outputs,
        true_boxes: torch.Tensor,
        true_labels: torch.Tensor
):
    predicted_boxes, scores, predicted_labels = decode_one_to_one(
        outputs
    )

    iou = aligned_iou(
        predicted_boxes,
        true_boxes
    )

    class_correct = (
            predicted_labels
            == true_labels
    )

    detected = (
            (iou >= 0.5)
            & class_correct
    )

    return (
        iou.mean().item(),
        detected.float().mean().item(),
        class_correct.float().mean().item(),
        scores.mean().item()
    )


def train_epoch(
        model: nn.Module,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device,
        epoch_index: int,
        epochs: int
):
    model.train()

    total_loss = 0.0
    total = 0

    one_to_one_weight = (
            0.5
            + 0.5
            * (
                epoch_index
                / max(epochs - 1, 1)
            )
    )

    for images, boxes, labels in tqdm(
            dataloader,
            desc="Training"
    ):
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss, _, _ = yolo26_loss(
            outputs=outputs,
            boxes=boxes,
            labels=labels,
            num_classes=model.num_classes,
            one_to_one_weight=one_to_one_weight
        )

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)

        total_loss += (
                loss.item()
                * batch_size
        )

        total += batch_size

    return total_loss / total


@torch.no_grad()
def evaluate(
        model: nn.Module,
        dataloader: DataLoader,
        device
):
    model.eval()

    total_loss = 0.0
    total_iou = 0.0
    total_accuracy50 = 0.0
    total_class_accuracy = 0.0
    total_score = 0.0
    total = 0

    for images, boxes, labels in tqdm(
            dataloader,
            desc="Validation"
    ):
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss, _, _ = yolo26_loss(
            outputs=outputs,
            boxes=boxes,
            labels=labels,
            num_classes=model.num_classes,
            one_to_one_weight=1.0
        )

        mean_iou, accuracy50, class_accuracy, mean_score = detection_metrics(
            outputs=outputs,
            true_boxes=boxes,
            true_labels=labels
        )

        batch_size = images.size(0)

        total_loss += (
                loss.item()
                * batch_size
        )

        total_iou += (
                mean_iou
                * batch_size
        )

        total_accuracy50 += (
                accuracy50
                * batch_size
        )

        total_class_accuracy += (
                class_accuracy
                * batch_size
        )

        total_score += (
                mean_score
                * batch_size
        )

        total += batch_size

    return (
        total_loss / total,
        total_iou / total,
        total_accuracy50 / total,
        total_class_accuracy / total,
        total_score / total
    )


def train_model(
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        device,
        epochs: int = 5,
        lr: float = 1e-3
):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr
    )

    for epoch in range(epochs):
        train_loss = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch_index=epoch,
            epochs=epochs
        )

        (
            test_loss,
            test_iou,
            test_accuracy50,
            test_class_accuracy,
            test_score
        ) = evaluate(
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
            f"Test loss: {test_loss:.4f}"
        )

        print(
            f"Test mean IoU: {test_iou:.4f}"
        )

        print(
            f"Test detection accuracy@0.5: {test_accuracy50:.4f}"
        )

        print(
            f"Test class accuracy: {test_class_accuracy:.4f}"
        )

        print(
            f"Test mean confidence: {test_score:.4f}"
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

    outputs = model(image)

    box, score, label = decode_one_to_one(
        outputs
    )

    return (
        box.squeeze(0).cpu(),
        score.squeeze(0).cpu(),
        label.squeeze(0).cpu()
    )


def count_parameters(
        model: nn.Module
) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
    )


def save_dataset_split_for_ultralytics(
        dataset: SyntheticDetectionDataset,
        root: Path,
        split: str
):
    image_dir = (
            root
            / "images"
            / split
    )

    label_dir = (
            root
            / "labels"
            / split
    )

    image_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    label_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    for index in range(len(dataset)):
        image, box, label = dataset[index]

        image_uint8 = (
                image.permute(
                    1,
                    2,
                    0
                ).numpy()
                * 255.0
        ).clip(
            0,
            255
        ).astype(
            np.uint8
        )

        image_path = (
                image_dir
                / f"{index:05d}.png"
        )

        Image.fromarray(
            image_uint8
        ).save(
            image_path
        )

        x1, y1, x2, y2 = box.tolist()

        center_x = (
                x1
                + x2
        ) / 2.0

        center_y = (
                y1
                + y2
        ) / 2.0

        width = x2 - x1
        height = y2 - y1

        label_path = (
                label_dir
                / f"{index:05d}.txt"
        )

        label_path.write_text(
            (
                f"{int(label.item())} "
                f"{center_x:.8f} "
                f"{center_y:.8f} "
                f"{width:.8f} "
                f"{height:.8f}\n"
            ),
            encoding="utf-8"
        )


def prepare_ultralytics_dataset(
        train_dataset: SyntheticDetectionDataset,
        test_dataset: SyntheticDetectionDataset,
        root: Path
) -> Path:
    save_dataset_split_for_ultralytics(
        dataset=train_dataset,
        root=root,
        split="train"
    )

    save_dataset_split_for_ultralytics(
        dataset=test_dataset,
        root=root,
        split="val"
    )

    yaml_path = (
            root
            / "dataset.yaml"
    )

    yaml_text = (
        f"path: {root.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: red_rectangle\n"
        "  1: green_rectangle\n"
        "  2: blue_rectangle\n"
    )

    yaml_path.write_text(
        yaml_text,
        encoding="utf-8"
    )

    return yaml_path


def evaluate_ultralytics_on_synthetic(
        model,
        dataset: SyntheticDetectionDataset,
        device,
        image_size: int
):
    device_argument = (
        0
        if device.type == "cuda"
        else "cpu"
    )

    total_iou = 0.0
    total_accuracy50 = 0.0
    total_class_accuracy = 0.0
    total_score = 0.0

    for index in tqdm(
            range(len(dataset)),
            desc="Ultralytics evaluation"
    ):
        image, true_box, true_label = dataset[index]

        image_uint8 = (
                image.permute(
                    1,
                    2,
                    0
                ).numpy()
                * 255.0
        ).clip(
            0,
            255
        ).astype(
            np.uint8
        )

        results = model.predict(
            source=image_uint8,
            imgsz=image_size,
            conf=0.001,
            device=device_argument,
            verbose=False,
            nms=False
        )

        result = results[0]

        if (
                result.boxes is None
                or len(result.boxes) == 0
        ):
            continue

        confidences = result.boxes.conf

        best_index = int(
            confidences.argmax().item()
        )

        predicted_box = result.boxes.xyxy[
            best_index
        ].detach().cpu().float()

        predicted_box[
            [0, 2]
        ] /= image_size

        predicted_box[
            [1, 3]
        ] /= image_size

        predicted_label = int(
            result.boxes.cls[
                best_index
            ].item()
        )

        score = float(
            confidences[
                best_index
            ].item()
        )

        iou = aligned_iou(
            predicted_box.unsqueeze(0),
            true_box.unsqueeze(0)
        ).item()

        class_correct = (
                predicted_label
                == int(true_label.item())
        )

        total_iou += iou
        total_class_accuracy += float(class_correct)
        total_accuracy50 += float(
            iou >= 0.5
            and class_correct
        )
        total_score += score

    total = len(dataset)

    return (
        total_iou / total,
        total_accuracy50 / total,
        total_class_accuracy / total,
        total_score / total
    )


def train_ultralytics_model(
        train_dataset: SyntheticDetectionDataset,
        test_dataset: SyntheticDetectionDataset,
        device,
        epochs: int,
        image_size: int
):
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError(
            "Install Ultralytics first: pip install -U ultralytics"
        ) from error

    dataset_root = Path(
        "synthetic_yolo26_dataset"
    ).resolve()

    yaml_path = prepare_ultralytics_dataset(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        root=dataset_root
    )

    device_argument = (
        0
        if device.type == "cuda"
        else "cpu"
    )

    framework_model = YOLO(
        "yolo26n.yaml"
    )

    framework_model.info()

    framework_model.train(
        data=str(yaml_path),
        epochs=epochs,
        imgsz=image_size,
        batch=32,
        device=device_argument,
        workers=0,
        optimizer="AdamW",
        lr0=1e-3,
        pretrained=False,
        plots=False,
        project="runs_yolo26_educational",
        name="ultralytics_yolo26n",
        exist_ok=True,
        verbose=False
    )

    best_path = framework_model.trainer.best

    trained_model = YOLO(
        str(best_path)
    )

    validation_metrics = trained_model.val(
        data=str(yaml_path),
        imgsz=image_size,
        batch=32,
        device=device_argument,
        workers=0,
        nms=False,
        plots=False,
        verbose=False
    )

    (
        mean_iou,
        accuracy50,
        class_accuracy,
        mean_score
    ) = evaluate_ultralytics_on_synthetic(
        model=trained_model,
        dataset=test_dataset,
        device=device,
        image_size=image_size
    )

    return {
        "model": trained_model,
        "map50": float(validation_metrics.box.map50),
        "map50_95": float(validation_metrics.box.map),
        "mean_iou": mean_iou,
        "accuracy50": accuracy50,
        "class_accuracy": class_accuracy,
        "mean_score": mean_score
    }


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

    image_size = 128
    num_classes = 3
    epochs = 5

    train_dataset = SyntheticDetectionDataset(
        n_samples=600,
        image_size=image_size,
        num_classes=num_classes,
        seed=42
    )

    test_dataset = SyntheticDetectionDataset(
        n_samples=300,
        image_size=image_size,
        num_classes=num_classes,
        seed=1337
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0
    )

    image, true_box, true_label = test_dataset[0]

    print(
        "\n"
        "======================================="
    )

    print(
        "CUSTOM EDUCATIONAL YOLO26"
    )

    print(
        "======================================="
    )

    custom_yolo = YOLO26(
        num_classes=num_classes,
        base_channels=8
    ).to(device)

    print(
        custom_yolo
    )

    print(
        "Custom parameters:",
        count_parameters(custom_yolo)
    )

    train_model(
        model=custom_yolo,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        epochs=epochs,
        lr=1e-3
    )

    (
        custom_test_loss,
        custom_test_iou,
        custom_test_accuracy50,
        custom_test_class_accuracy,
        custom_test_mean_score
    ) = evaluate(
        model=custom_yolo,
        dataloader=test_loader,
        device=device
    )

    predicted_box, predicted_score, predicted_label = inference(
        model=custom_yolo,
        image=image,
        device=device
    )

    custom_outputs = custom_yolo(
        image.unsqueeze(0).to(device)
    )

    (
        sample_iou,
        sample_accuracy50,
        sample_class_accuracy,
        sample_mean_score
    ) = detection_metrics(
        outputs=custom_outputs,
        true_boxes=true_box.unsqueeze(0).to(device),
        true_labels=true_label.unsqueeze(0).to(device)
    )

    print(
        "\nCustom YOLO26 inference"
    )

    print(
        "Image shape:",
        image.shape
    )

    print(
        "True box:",
        true_box.tolist()
    )

    print(
        "Predicted box:",
        predicted_box.tolist()
    )

    print(
        "True class:",
        int(true_label.item())
    )

    print(
        "Predicted class:",
        int(predicted_label.item())
    )

    print(
        "Confidence:",
        round(
            float(predicted_score.item()),
            4
        )
    )

    print(
        "IoU:",
        round(
            sample_iou,
            4
        )
    )

    print(
        "\n"
        "======================================="
    )

    print(
        "ULTRALYTICS YOLO26N"
    )

    print(
        "======================================="
    )

    framework_metrics = train_ultralytics_model(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        device=device,
        epochs=epochs,
        image_size=image_size
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
        f"Custom YOLO26 mean IoU:             {custom_test_iou:.4f}"
    )

    print(
        f"Custom YOLO26 Acc@0.5:              {custom_test_accuracy50:.4f}"
    )

    print(
        f"Custom YOLO26 class accuracy:       {custom_test_class_accuracy:.4f}"
    )

    print()

    print(
        f"Ultralytics YOLO26n mean IoU:       {framework_metrics['mean_iou']:.4f}"
    )

    print(
        f"Ultralytics YOLO26n Acc@0.5:        {framework_metrics['accuracy50']:.4f}"
    )

    print(
        f"Ultralytics YOLO26n class accuracy: {framework_metrics['class_accuracy']:.4f}"
    )

    print(
        f"Ultralytics YOLO26n mAP50:         {framework_metrics['map50']:.4f}"
    )

    print(
        f"Ultralytics YOLO26n mAP50-95:      {framework_metrics['map50_95']:.4f}"
    )


if __name__ == "__main__":
    main()
