import random

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import retinanet_resnet50_fpn
from torchvision.ops import batched_nms, box_iou
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
            padding: int | None = None
    ):
        super().__init__()

        if padding is None:
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

class ResidualBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int = 1
    ):
        super().__init__()

        self.conv1 = ConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                out_channels
            )
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

        self.relu = nn.ReLU(
            inplace=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)

        x = self.conv1(x)
        x = self.conv2(x)

        x = x + identity
        x = self.relu(x)

        return x

class RetinaNetBackbone(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            base_channels: int = 32
    ):
        super().__init__()

        self.stem = nn.Sequential(
            ConvBlock(
                in_channels=in_channels,
                out_channels=base_channels,
                kernel_size=7,
                stride=2,
                padding=3
            ),

            nn.MaxPool2d(
                kernel_size=3,
                stride=2,
                padding=1
            )
        )

        self.stage2 = nn.Sequential(
            ResidualBlock(
                in_channels=base_channels,
                out_channels=base_channels,
                stride=1
            ),

            ResidualBlock(
                in_channels=base_channels,
                out_channels=base_channels,
                stride=1
            )
        )

        self.stage3 = nn.Sequential(
            ResidualBlock(
                in_channels=base_channels,
                out_channels=base_channels * 2,
                stride=2
            ),

            ResidualBlock(
                in_channels=base_channels * 2,
                out_channels=base_channels * 2,
                stride=1
            )
        )

        self.stage4 = nn.Sequential(
            ResidualBlock(
                in_channels=base_channels * 2,
                out_channels=base_channels * 4,
                stride=2
            ),

            ResidualBlock(
                in_channels=base_channels * 4,
                out_channels=base_channels * 4,
                stride=1
            )
        )

        self.stage5 = nn.Sequential(
            ResidualBlock(
                in_channels=base_channels * 4,
                out_channels=base_channels * 8,
                stride=2
            ),

            ResidualBlock(
                in_channels=base_channels * 8,
                out_channels=base_channels * 8,
                stride=1
            )
        )

    def forward(
            self,
            x: torch.Tensor
    ):
        x = self.stem(x)
        x = self.stage2(x)

        c3 = self.stage3(x)
        c4 = self.stage4(c3)
        c5 = self.stage5(c4)

        return c3, c4, c5

class FPN(nn.Module):
    def __init__(
            self,
            c3_channels: int,
            c4_channels: int,
            c5_channels: int,
            fpn_channels: int = 128
    ):
        super().__init__()

        self.lateral3 = nn.Conv2d(
            in_channels=c3_channels,
            out_channels=fpn_channels,
            kernel_size=1
        )

        self.lateral4 = nn.Conv2d(
            in_channels=c4_channels,
            out_channels=fpn_channels,
            kernel_size=1
        )

        self.lateral5 = nn.Conv2d(
            in_channels=c5_channels,
            out_channels=fpn_channels,
            kernel_size=1
        )

        self.output3 = nn.Conv2d(
            in_channels=fpn_channels,
            out_channels=fpn_channels,
            kernel_size=3,
            padding=1
        )

        self.output4 = nn.Conv2d(
            in_channels=fpn_channels,
            out_channels=fpn_channels,
            kernel_size=3,
            padding=1
        )

        self.output5 = nn.Conv2d(
            in_channels=fpn_channels,
            out_channels=fpn_channels,
            kernel_size=3,
            padding=1
        )

        self.p6 = nn.Conv2d(
            in_channels=c5_channels,
            out_channels=fpn_channels,
            kernel_size=3,
            stride=2,
            padding=1
        )

        self.p7 = nn.Conv2d(
            in_channels=fpn_channels,
            out_channels=fpn_channels,
            kernel_size=3,
            stride=2,
            padding=1
        )

    def forward(
            self,
            c3: torch.Tensor,
            c4: torch.Tensor,
            c5: torch.Tensor
    ):
        p5_lateral = self.lateral5(c5)

        p4_lateral = (
                self.lateral4(c4)
                + F.interpolate(
            p5_lateral,
            size=c4.shape[-2:],
            mode="nearest"
        )
        )

        p3_lateral = (
                self.lateral3(c3)
                + F.interpolate(
            p4_lateral,
            size=c3.shape[-2:],
            mode="nearest"
        )
        )

        p3 = self.output3(p3_lateral)
        p4 = self.output4(p4_lateral)
        p5 = self.output5(p5_lateral)

        p6 = self.p6(c5)
        p7 = self.p7(F.relu(p6))

        return [
            p3,
            p4,
            p5,
            p6,
            p7
        ]

class ClassificationSubnet(nn.Module):
    def __init__(
            self,
            in_channels: int,
            num_anchors: int,
            num_classes: int
    ):
        super().__init__()

        layers = []

        for _ in range(4):
            layers.extend([
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=3,
                    padding=1
                ),

                nn.ReLU(
                    inplace=True
                )
            ])

        self.conv = nn.Sequential(
            *layers
        )

        self.output = nn.Conv2d(
            in_channels=in_channels,
            out_channels=(num_anchors * num_classes),
            kernel_size=3,
            padding=1
        )

        prior_probability = 0.01

        bias = -np.log(
            (1.0 - prior_probability)
            / prior_probability
        )

        nn.init.constant_(
            self.output.bias,
            bias
        )

    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:
        x = self.conv(x)
        return self.output(x)

class RegressionSubnet(nn.Module):
    def __init__(
            self,
            in_channels: int,
            num_anchors: int
    ):
        super().__init__()

        layers = []

        for _ in range(4):
            layers.extend([
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=3,
                    padding=1
                ),

                nn.ReLU(
                    inplace=True
                )
            ])

        self.conv = nn.Sequential(
            *layers
        )

        self.output = nn.Conv2d(
            in_channels=in_channels,
            out_channels=(num_anchors * 4),
            kernel_size=3,
            padding=1
        )

    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:
        x = self.conv(x)
        return self.output(x)

class AnchorGenerator:
    def __init__(
            self,
            sizes=(16, 32, 64, 128, 256),
            aspect_ratios=(0.5, 1.0, 2.0),
            scales=(1.0, 2 ** (1 / 3), 2 ** (2 / 3))
    ):
        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        self.scales = scales

        self.num_anchors = (
                len(aspect_ratios)
                * len(scales)
        )

    def _base_anchors(
            self,
            size: float,
            device
    ) -> torch.Tensor:
        anchors = []

        for ratio in self.aspect_ratios:
            for scale in self.scales:
                area = (
                        size
                        * scale
                ) ** 2

                width = np.sqrt(
                    area / ratio
                )

                height = (
                        width
                        * ratio
                )

                anchors.append([
                    -width / 2,
                    -height / 2,
                    width / 2,
                    height / 2
                ])

        return torch.tensor(
            anchors,
            dtype=torch.float32,
            device=device
        )

    def generate(
            self,
            feature_maps,
            image_height: int,
            image_width: int
    ) -> torch.Tensor:
        all_anchors = []

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

            base_anchors = self._base_anchors(
                size=self.sizes[level],
                device=feature.device
            )

            shifts_x = (
                    torch.arange(
                        width,
                        device=feature.device,
                        dtype=torch.float32
                    )
                    + 0.5
            ) * stride_x

            shifts_y = (
                    torch.arange(
                        height,
                        device=feature.device,
                        dtype=torch.float32
                    )
                    + 0.5
            ) * stride_y

            shift_y, shift_x = torch.meshgrid(
                shifts_y,
                shifts_x,
                indexing="ij"
            )

            shifts = torch.stack(
                [
                    shift_x,
                    shift_y,
                    shift_x,
                    shift_y
                ],
                dim=-1
            ).reshape(
                -1,
                4
            )

            anchors = (
                    shifts[:, None, :]
                    + base_anchors[None, :, :]
            ).reshape(
                -1,
                4
            )

            all_anchors.append(
                anchors
            )

        return torch.cat(
            all_anchors,
            dim=0
        )

class RetinaNet(nn.Module):
    def __init__(
            self,
            num_classes: int = 3,
            base_channels: int = 32,
            fpn_channels: int = 128
    ):
        super().__init__()

        self.num_classes = num_classes

        self.backbone = RetinaNetBackbone(
            in_channels=3,
            base_channels=base_channels
        )

        self.fpn = FPN(
            c3_channels=base_channels * 2,
            c4_channels=base_channels * 4,
            c5_channels=base_channels * 8,
            fpn_channels=fpn_channels
        )

        self.anchor_generator = AnchorGenerator()

        num_anchors = (
            self.anchor_generator.num_anchors
        )

        self.classification_head = ClassificationSubnet(
            in_channels=fpn_channels,
            num_anchors=num_anchors,
            num_classes=num_classes
        )

        self.regression_head = RegressionSubnet(
            in_channels=fpn_channels,
            num_anchors=num_anchors
        )

    def forward(
            self,
            images: torch.Tensor
    ):
        c3, c4, c5 = self.backbone(
            images
        )

        features = self.fpn(
            c3,
            c4,
            c5
        )

        classification = []
        regression = []

        for feature in features:
            classification.append(
                self.classification_head(feature)
            )

            regression.append(
                self.regression_head(feature)
            )

        anchors = self.anchor_generator.generate(
            feature_maps=features,
            image_height=images.shape[-2],
            image_width=images.shape[-1]
        )

        return classification, regression, anchors

class SyntheticDetectionDataset(Dataset):
    """
    Each image contains exactly one colored rectangle.

    Class 0 -> red rectangle
    Class 1 -> green rectangle
    Class 2 -> blue rectangle

    Boxes are returned in xyxy pixel format.
    """

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
        targets = []

        for index in range(n_samples):
            image = torch.randn(
                3,
                image_size,
                image_size,
                generator=generator
            ) * 0.04

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
                    high=(image_size - rect_height),
                    size=(1,),
                    generator=generator
                ).item()
            )

            x1 = int(
                torch.randint(
                    low=0,
                    high=(image_size - rect_width),
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

            color = torch.zeros(3)
            color[label] = 1.0

            image[
                :,
                y1:y2,
                x1:x2
            ] = color[:, None, None]

            target = {
                "boxes": torch.tensor(
                    [[x1, y1, x2, y2]],
                    dtype=torch.float32
                ),

                "labels": torch.tensor(
                    [label],
                    dtype=torch.long
                ),

                "image_id": torch.tensor(
                    index,
                    dtype=torch.long
                )
            }

            images.append(
                image
            )

            targets.append(
                target
            )

        self.images = images
        self.targets = targets

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
            self.targets[index]
        )


def detection_collate_fn(
        batch
):
    images = []
    targets = []

    for image, target in batch:
        images.append(image)
        targets.append(target)

    return images, targets

def encode_boxes(
        anchors: torch.Tensor,
        boxes: torch.Tensor
) -> torch.Tensor:
    anchor_widths = (
            anchors[:, 2]
            - anchors[:, 0]
    ).clamp(min=1e-6)

    anchor_heights = (
            anchors[:, 3]
            - anchors[:, 1]
    ).clamp(min=1e-6)

    anchor_centers_x = (
            anchors[:, 0]
            + 0.5 * anchor_widths
    )

    anchor_centers_y = (
            anchors[:, 1]
            + 0.5 * anchor_heights
    )

    box_widths = (
            boxes[:, 2]
            - boxes[:, 0]
    ).clamp(min=1e-6)

    box_heights = (
            boxes[:, 3]
            - boxes[:, 1]
    ).clamp(min=1e-6)

    box_centers_x = (
            boxes[:, 0]
            + 0.5 * box_widths
    )

    box_centers_y = (
            boxes[:, 1]
            + 0.5 * box_heights
    )

    dx = (
            box_centers_x
            - anchor_centers_x
    ) / anchor_widths

    dy = (
            box_centers_y
            - anchor_centers_y
    ) / anchor_heights

    dw = torch.log(
        box_widths / anchor_widths
    )

    dh = torch.log(
        box_heights / anchor_heights
    )

    return torch.stack(
        [dx, dy, dw, dh],
        dim=1
    )


def decode_boxes(
        anchors: torch.Tensor,
        regression: torch.Tensor
) -> torch.Tensor:
    anchor_widths = (
            anchors[:, 2]
            - anchors[:, 0]
    ).clamp(min=1e-6)

    anchor_heights = (
            anchors[:, 3]
            - anchors[:, 1]
    ).clamp(min=1e-6)

    anchor_centers_x = (
            anchors[:, 0]
            + 0.5 * anchor_widths
    )

    anchor_centers_y = (
            anchors[:, 1]
            + 0.5 * anchor_heights
    )

    dx = regression[:, 0]
    dy = regression[:, 1]
    dw = regression[:, 2].clamp(
        min=-4.0,
        max=4.0
    )
    dh = regression[:, 3].clamp(
        min=-4.0,
        max=4.0
    )

    centers_x = (
            dx * anchor_widths
            + anchor_centers_x
    )

    centers_y = (
            dy * anchor_heights
            + anchor_centers_y
    )

    widths = (
            torch.exp(dw)
            * anchor_widths
    )

    heights = (
            torch.exp(dh)
            * anchor_heights
    )

    x1 = centers_x - widths / 2
    y1 = centers_y - heights / 2
    x2 = centers_x + widths / 2
    y2 = centers_y + heights / 2

    return torch.stack(
        [x1, y1, x2, y2],
        dim=1
    )


def flatten_classification(
        predictions,
        num_classes: int,
        num_anchors: int
) -> torch.Tensor:
    flattened = []

    for prediction in predictions:
        batch_size, _, height, width = (
            prediction.shape
        )

        prediction = prediction.view(
            batch_size,
            num_anchors,
            num_classes,
            height,
            width
        )

        prediction = prediction.permute(
            0,
            3,
            4,
            1,
            2
        ).reshape(
            batch_size,
            -1,
            num_classes
        )

        flattened.append(
            prediction
        )

    return torch.cat(
        flattened,
        dim=1
    )


def flatten_regression(
        predictions,
        num_anchors: int
) -> torch.Tensor:
    flattened = []

    for prediction in predictions:
        batch_size, _, height, width = (
            prediction.shape
        )

        prediction = prediction.view(
            batch_size,
            num_anchors,
            4,
            height,
            width
        )

        prediction = prediction.permute(
            0,
            3,
            4,
            1,
            2
        ).reshape(
            batch_size,
            -1,
            4
        )

        flattened.append(
            prediction
        )

    return torch.cat(
        flattened,
        dim=1
    )


def sigmoid_focal_loss(
        logits: torch.Tensor,
        targets: torch.Tensor,
        alpha: float = 0.25,
        gamma: float = 2.0
) -> torch.Tensor:
    probabilities = torch.sigmoid(
        logits
    )

    ce_loss = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none"
    )

    p_t = (
            probabilities * targets
            + (1.0 - probabilities) * (1.0 - targets)
    )

    focal_factor = (
            1.0
            - p_t
    ) ** gamma

    alpha_factor = (
            alpha * targets
            + (1.0 - alpha) * (1.0 - targets)
    )

    return (
            alpha_factor
            * focal_factor
            * ce_loss
    )


class RetinaNetLoss(nn.Module):
    def __init__(
            self,
            num_classes: int,
            num_anchors: int,
            positive_iou_threshold: float = 0.5,
            negative_iou_threshold: float = 0.4
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.positive_iou_threshold = positive_iou_threshold
        self.negative_iou_threshold = negative_iou_threshold

    def forward(
            self,
            classification,
            regression,
            anchors: torch.Tensor,
            targets
    ):
        cls_logits = flatten_classification(
            predictions=classification,
            num_classes=self.num_classes,
            num_anchors=self.num_anchors
        )

        bbox_regression = flatten_regression(
            predictions=regression,
            num_anchors=self.num_anchors
        )

        total_cls_loss = torch.tensor(
            0.0,
            device=anchors.device
        )

        total_box_loss = torch.tensor(
            0.0,
            device=anchors.device
        )

        batch_size = cls_logits.size(0)

        for batch_index in range(batch_size):
            gt_boxes = targets[batch_index]["boxes"].to(
                anchors.device
            )

            gt_labels = targets[batch_index]["labels"].to(
                anchors.device
            )

            ious = box_iou(
                anchors,
                gt_boxes
            )

            max_iou, matched_gt = ious.max(
                dim=1
            )

            positive = (
                    max_iou
                    >= self.positive_iou_threshold
            )

            negative = (
                    max_iou
                    < self.negative_iou_threshold
            )

            # Guarantee that every ground-truth object has at least
            # one positive anchor.
            best_anchor_for_gt = ious.argmax(
                dim=0
            )

            positive[
                best_anchor_for_gt
            ] = True

            negative[
                best_anchor_for_gt
            ] = False

            valid = (
                    positive
                    | negative
            )

            cls_targets = torch.zeros_like(
                cls_logits[batch_index]
            )

            if positive.any():
                positive_labels = gt_labels[
                    matched_gt[positive]
                ]

                cls_targets[
                    positive,
                    positive_labels
                ] = 1.0

            cls_loss = sigmoid_focal_loss(
                logits=cls_logits[batch_index][valid],
                targets=cls_targets[valid]
            ).sum()

            num_positive = max(
                int(positive.sum().item()),
                1
            )

            cls_loss = (
                    cls_loss
                    / num_positive
            )

            total_cls_loss = (
                    total_cls_loss
                    + cls_loss
            )

            if positive.any():
                matched_boxes = gt_boxes[
                    matched_gt[positive]
                ]

                regression_targets = encode_boxes(
                    anchors=anchors[positive],
                    boxes=matched_boxes
                )

                box_loss = F.smooth_l1_loss(
                    bbox_regression[batch_index][positive],
                    regression_targets,
                    reduction="sum",
                    beta=1.0 / 9.0
                )

                box_loss = (
                        box_loss
                        / num_positive
                )

                total_box_loss = (
                        total_box_loss
                        + box_loss
                )

        total_cls_loss = (
                total_cls_loss
                / batch_size
        )

        total_box_loss = (
                total_box_loss
                / batch_size
        )

        total_loss = (
                total_cls_loss
                + total_box_loss
        )

        return (
            total_loss,
            total_cls_loss,
            total_box_loss
        )


@torch.no_grad()
def inference(
        model: RetinaNet,
        image: torch.Tensor,
        device,
        score_threshold: float = 0.20,
        nms_threshold: float = 0.5,
        max_detections: int = 100
):
    model.eval()

    image_batch = image.unsqueeze(
        dim=0
    ).to(
        device
    )

    classification, regression, anchors = model(
        image_batch
    )

    cls_logits = flatten_classification(
        predictions=classification,
        num_classes=model.num_classes,
        num_anchors=model.anchor_generator.num_anchors
    )[0]

    bbox_regression = flatten_regression(
        predictions=regression,
        num_anchors=model.anchor_generator.num_anchors
    )[0]

    probabilities = torch.sigmoid(
        cls_logits
    )

    scores, labels = probabilities.max(
        dim=1
    )

    keep = (
            scores
            >= score_threshold
    )

    if not keep.any():
        return {
            "boxes": torch.empty((0, 4)),
            "labels": torch.empty((0,), dtype=torch.long),
            "scores": torch.empty((0,))
        }

    scores = scores[keep]
    labels = labels[keep]
    regression_keep = bbox_regression[keep]
    anchors_keep = anchors[keep]

    boxes = decode_boxes(
        anchors=anchors_keep,
        regression=regression_keep
    )

    boxes[:, 0::2] = boxes[:, 0::2].clamp(
        0,
        image.shape[-1]
    )

    boxes[:, 1::2] = boxes[:, 1::2].clamp(
        0,
        image.shape[-2]
    )

    keep_indices = batched_nms(
        boxes=boxes,
        scores=scores,
        idxs=labels,
        iou_threshold=nms_threshold
    )

    keep_indices = keep_indices[
        :max_detections
    ]

    return {
        "boxes": boxes[keep_indices].cpu(),
        "labels": labels[keep_indices].cpu(),
        "scores": scores[keep_indices].cpu()
    }


def single_object_metrics(
        prediction,
        target
):
    if prediction["boxes"].numel() == 0:
        return 0.0, 0.0, 0.0

    best_index = prediction["scores"].argmax()

    predicted_box = prediction["boxes"][
        best_index
    ].unsqueeze(0)

    predicted_label = int(
        prediction["labels"][best_index].item()
    )

    true_box = target["boxes"]
    true_label = int(
        target["labels"][0].item()
    )

    iou = box_iou(
        predicted_box,
        true_box
    )[0, 0].item()

    class_correct = float(
        predicted_label == true_label
    )

    detection_correct = float(
        iou >= 0.5
        and predicted_label == true_label
    )

    return (
        iou,
        class_correct,
        detection_correct
    )


@torch.no_grad()
def evaluate_custom_detector(
        model: RetinaNet,
        dataloader: DataLoader,
        device
):
    model.eval()

    total_iou = 0.0
    total_class_accuracy = 0.0
    total_detection_accuracy = 0.0
    total = 0

    for images, targets in tqdm(
            dataloader,
            desc="Custom validation"
    ):
        for image, target in zip(
                images,
                targets
        ):
            prediction = inference(
                model=model,
                image=image,
                device=device
            )

            iou, class_accuracy, detection_accuracy = (
                single_object_metrics(
                    prediction=prediction,
                    target=target
                )
            )

            total_iou += iou
            total_class_accuracy += class_accuracy
            total_detection_accuracy += detection_accuracy
            total += 1

    return (
        total_iou / total,
        total_class_accuracy / total,
        total_detection_accuracy / total
    )

def train_epoch(
        model: RetinaNet,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: RetinaNetLoss,
        device
):
    model.train()

    total_loss = 0.0
    total_cls_loss = 0.0
    total_box_loss = 0.0
    total = 0

    for images, targets in tqdm(
            dataloader,
            desc="Training custom RetinaNet"
    ):
        images = torch.stack(
            images
        ).to(
            device
        )

        targets_device = []

        for target in targets:
            targets_device.append({
                "boxes": target["boxes"].to(device),
                "labels": target["labels"].to(device)
            })

        optimizer.zero_grad()

        classification, regression, anchors = model(
            images
        )

        loss, cls_loss, box_loss = criterion(
            classification=classification,
            regression=regression,
            anchors=anchors,
            targets=targets_device
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=10.0
        )

        optimizer.step()

        batch_size = images.size(0)

        total_loss += (
                loss.item()
                * batch_size
        )

        total_cls_loss += (
                cls_loss.item()
                * batch_size
        )

        total_box_loss += (
                box_loss.item()
                * batch_size
        )

        total += batch_size

    return (
        total_loss / total,
        total_cls_loss / total,
        total_box_loss / total
    )


def train_model(
        model: RetinaNet,
        train_loader: DataLoader,
        test_loader: DataLoader,
        device,
        epochs: int = 5,
        lr: float = 1e-3
):
    criterion = RetinaNetLoss(
        num_classes=model.num_classes,
        num_anchors=model.anchor_generator.num_anchors
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4
    )

    for epoch in range(epochs):
        train_loss, cls_loss, box_loss = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device
        )

        mean_iou, class_accuracy, detection_accuracy = (
            evaluate_custom_detector(
                model=model,
                dataloader=test_loader,
                device=device
            )
        )

        print(
            f"Epoch {epoch + 1}/{epochs}"
        )

        print(
            f"Train loss: {train_loss:.4f}"
        )

        print(
            f"Classification loss: {cls_loss:.4f}"
        )

        print(
            f"Box loss: {box_loss:.4f}"
        )

        print(
            f"Validation mean IoU: {mean_iou:.4f}"
        )

        print(
            f"Validation class accuracy: {class_accuracy:.4f}"
        )

        print(
            f"Validation detection accuracy@0.5: {detection_accuracy:.4f}"
        )

        print()


def convert_targets_for_torchvision(
        targets,
        device
):
    converted = []

    for target in targets:

        converted.append({
            "boxes": target["boxes"].to(device),
            "labels": (
                    target["labels"].to(device)
                    + 1
            )
        })

    return converted


def train_torchvision_epoch(
        model: nn.Module,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device
):
    model.train()

    total_loss = 0.0
    total_classification_loss = 0.0
    total_box_loss = 0.0
    total = 0

    for images, targets in tqdm(
            dataloader,
            desc="Training torchvision RetinaNet"
    ):
        images = [
            image.to(device)
            for image in images
        ]

        converted_targets = convert_targets_for_torchvision(
            targets=targets,
            device=device
        )

        optimizer.zero_grad()

        loss_dict = model(
            images,
            converted_targets
        )

        loss = sum(
            loss_dict.values()
        )

        loss.backward()
        optimizer.step()

        batch_size = len(images)

        classification_loss = loss_dict.get(
            "classification",
            torch.tensor(0.0, device=device)
        )

        box_loss = loss_dict.get(
            "bbox_regression",
            torch.tensor(0.0, device=device)
        )

        total_loss += (
                loss.item()
                * batch_size
        )

        total_classification_loss += (
                classification_loss.item()
                * batch_size
        )

        total_box_loss += (
                box_loss.item()
                * batch_size
        )

        total += batch_size

    return (
        total_loss / total,
        total_classification_loss / total,
        total_box_loss / total
    )


@torch.no_grad()
def torchvision_inference(
        model: nn.Module,
        image: torch.Tensor,
        device
):
    model.eval()

    prediction = model([
        image.to(device)
    ])[0]

    return {
        "boxes": prediction["boxes"].cpu(),
        "labels": (
                prediction["labels"].cpu()
                - 1
        ),
        "scores": prediction["scores"].cpu()
    }


@torch.no_grad()
def evaluate_torchvision_detector(
        model: nn.Module,
        dataloader: DataLoader,
        device
):
    model.eval()

    total_iou = 0.0
    total_class_accuracy = 0.0
    total_detection_accuracy = 0.0
    total = 0

    for images, targets in tqdm(
            dataloader,
            desc="Torchvision validation"
    ):
        for image, target in zip(
                images,
                targets
        ):
            prediction = torchvision_inference(
                model=model,
                image=image,
                device=device
            )

            iou, class_accuracy, detection_accuracy = (
                single_object_metrics(
                    prediction=prediction,
                    target=target
                )
            )

            total_iou += iou
            total_class_accuracy += class_accuracy
            total_detection_accuracy += detection_accuracy
            total += 1

    return (
        total_iou / total,
        total_class_accuracy / total,
        total_detection_accuracy / total
    )


def train_torchvision_model(
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        device,
        epochs: int = 2,
        lr: float = 1e-4
):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4
    )

    for epoch in range(epochs):
        train_loss, cls_loss, box_loss = train_torchvision_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device
        )

        mean_iou, class_accuracy, detection_accuracy = (
            evaluate_torchvision_detector(
                model=model,
                dataloader=test_loader,
                device=device
            )
        )

        print(
            f"Epoch {epoch + 1}/{epochs}"
        )

        print(
            f"Train loss: {train_loss:.4f}"
        )

        print(
            f"Classification loss: {cls_loss:.4f}"
        )

        print(
            f"Box loss: {box_loss:.4f}"
        )

        print(
            f"Validation mean IoU: {mean_iou:.4f}"
        )

        print(
            f"Validation class accuracy: {class_accuracy:.4f}"
        )

        print(
            f"Validation detection accuracy@0.5: {detection_accuracy:.4f}"
        )

        print()


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

    image_size = 128
    num_classes = 3

    train_dataset = SyntheticDetectionDataset(
        n_samples=300,
        image_size=image_size,
        num_classes=num_classes,
        seed=42
    )

    test_dataset = SyntheticDetectionDataset(
        n_samples=100,
        image_size=image_size,
        num_classes=num_classes,
        seed=1337
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=detection_collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=detection_collate_fn
    )

    image, target = test_dataset[0]

    print(
        "\n"
        "======================================="
    )

    print(
        "CUSTOM EDUCATIONAL RETINANET"
    )

    print(
        "======================================="
    )

    custom_retinanet = RetinaNet(
        num_classes=num_classes,
        base_channels=32,
        fpn_channels=128
    ).to(
        device
    )

    print(
        custom_retinanet
    )

    print(
        "Trainable parameters:",
        count_parameters(custom_retinanet)
    )

    # Forward-shape check before training.
    with torch.no_grad():
        classification, regression, anchors = custom_retinanet(
            image.unsqueeze(0).to(device)
        )

    print(
        "\nFeature pyramid prediction shapes:"
    )

    for level, (cls, box) in enumerate(
            zip(classification, regression),
            start=3
    ):
        print(
            f"P{level}: cls={tuple(cls.shape)}, box={tuple(box.shape)}"
        )

    print(
        "Anchors shape:",
        anchors.shape
    )

    train_model(
        model=custom_retinanet,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        epochs=5,
        lr=1e-3
    )

    custom_prediction = inference(
        model=custom_retinanet,
        image=image,
        device=device
    )

    print(
        "\nCustom RetinaNet inference"
    )

    print(
        "Image shape:",
        image.shape
    )

    print(
        "True box:",
        target["boxes"]
    )

    print(
        "True class:",
        int(target["labels"][0].item())
    )

    print(
        "Predicted boxes:",
        custom_prediction["boxes"][:5]
    )

    print(
        "Predicted labels:",
        custom_prediction["labels"][:5]
    )

    print(
        "Predicted scores:",
        custom_prediction["scores"][:5]
    )

    custom_mean_iou, custom_class_accuracy, custom_detection_accuracy = (
        evaluate_custom_detector(
            model=custom_retinanet,
            dataloader=test_loader,
            device=device
        )
    )

    print(
        "\n"
        "======================================="
    )

    print(
        "TORCHVISION RETINANET RESNET50 FPN"
    )

    print(
        "======================================="
    )

    framework_retinanet = retinanet_resnet50_fpn(
        weights=None,
        weights_backbone=None,
        num_classes=(num_classes + 1),
        min_size=image_size,
        max_size=image_size,
        score_thresh=0.20
    ).to(
        device
    )

    print(
        framework_retinanet
    )

    print(
        "Trainable parameters:",
        count_parameters(framework_retinanet)
    )

    train_torchvision_model(
        model=framework_retinanet,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        epochs=2,
        lr=1e-4
    )

    framework_prediction = torchvision_inference(
        model=framework_retinanet,
        image=image,
        device=device
    )

    print(
        "\nTorchvision RetinaNet inference"
    )

    print(
        "Predicted boxes:",
        framework_prediction["boxes"][:5]
    )

    print(
        "Predicted labels:",
        framework_prediction["labels"][:5]
    )

    print(
        "Predicted scores:",
        framework_prediction["scores"][:5]
    )

    framework_mean_iou, framework_class_accuracy, framework_detection_accuracy = (
        evaluate_torchvision_detector(
            model=framework_retinanet,
            dataloader=test_loader,
            device=device
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
        f"Custom RetinaNet mean IoU:       {custom_mean_iou:.4f}"
    )

    print(
        f"Torchvision RetinaNet mean IoU:  {framework_mean_iou:.4f}"
    )

    print()

    print(
        f"Custom class accuracy:           {custom_class_accuracy:.4f}"
    )

    print(
        f"Torchvision class accuracy:      {framework_class_accuracy:.4f}"
    )

    print()

    print(
        f"Custom detection accuracy@0.5:   {custom_detection_accuracy:.4f}"
    )

    print(
        f"Torchvision detection accuracy:  {framework_detection_accuracy:.4f}"
    )


if __name__ == "__main__":
    main()