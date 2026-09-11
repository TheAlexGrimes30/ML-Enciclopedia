import random

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import fasterrcnn_resnet50_fpn
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

        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)

        x = self.conv1(x)
        x = self.conv2(x)

        x = x + identity
        x = self.activation(x)

        return x

class FasterRCNNBackbone(nn.Module):
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
            )
        )

        self.stage3 = nn.Sequential(
            ResidualBlock(
                in_channels=64,
                out_channels=128,
                stride=2
            ),

            ResidualBlock(
                in_channels=128,
                out_channels=128,
                stride=1
            )
        )

        self.stage4 = nn.Sequential(
            ResidualBlock(
                in_channels=128,
                out_channels=192,
                stride=2
            ),

            ResidualBlock(
                in_channels=192,
                out_channels=192,
                stride=1
            )
        )

        self.stage5 = nn.Sequential(
            ResidualBlock(
                in_channels=192,
                out_channels=256,
                stride=2
            ),

            ResidualBlock(
                in_channels=256,
                out_channels=256,
                stride=1
            )
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.stem(x)

        c3 = self.stage3(x)
        c4 = self.stage4(c3)
        c5 = self.stage5(c4)

        return [
            c3,
            c4,
            c5
        ]

class FPN(nn.Module):
    def __init__(
            self,
            in_channels=(128, 192, 256),
            out_channels: int = 128
    ):
        super().__init__()

        self.lateral3 = nn.Conv2d(
            in_channels=in_channels[0],
            out_channels=out_channels,
            kernel_size=1
        )

        self.lateral4 = nn.Conv2d(
            in_channels=in_channels[1],
            out_channels=out_channels,
            kernel_size=1
        )

        self.lateral5 = nn.Conv2d(
            in_channels=in_channels[2],
            out_channels=out_channels,
            kernel_size=1
        )

        self.output3 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1
        )

        self.output4 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1
        )

        self.output5 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1
        )

    def forward(self, features):
        c3, c4, c5 = features

        p5 = self.lateral5(c5)

        p4 = (
                self.lateral4(c4)
                + F.interpolate(
            p5,
            size=c4.shape[-2:],
            mode="nearest"
        )
        )

        p3 = (
                self.lateral3(c3)
                + F.interpolate(
            p4,
            size=c3.shape[-2:],
            mode="nearest"
        )
        )

        p3 = self.output3(p3)
        p4 = self.output4(p4)
        p5 = self.output5(p5)

        return [
            p3,
            p4,
            p5
        ]

class AnchorGenerator:
    def __init__(
            self,
            sizes=(16, 32, 64),
            aspect_ratios=(0.5, 1.0, 2.0)
    ):
        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        self.num_anchors = len(aspect_ratios)

    def _base_anchors(
            self,
            size: float,
            device
    ) -> torch.Tensor:
        anchors = []

        area = float(size * size)

        for ratio in self.aspect_ratios:
            width = np.sqrt(
                area
                / ratio
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
    ):
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
                    ) + 0.5) * stride_x

            shifts_y = (
                    torch.arange(
                    height,
                    device=feature.device,
                    dtype=torch.float32) + 0.5
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

class RPNHead(nn.Module):
    def __init__(
            self,
            channels: int = 128,
            num_anchors: int = 3
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            padding=1
        )

        self.objectness = nn.Conv2d(
            in_channels=channels,
            out_channels=num_anchors,
            kernel_size=1
        )

        self.box_regression = nn.Conv2d(
            in_channels=channels,
            out_channels=(
                    num_anchors
                    * 4
            ),
            kernel_size=1
        )

        for layer in [
            self.conv,
            self.objectness,
            self.box_regression
        ]:
            nn.init.normal_(
                layer.weight,
                std=0.01
            )

            nn.init.constant_(
                layer.bias,
                0.0
            )

    def forward(
            self,
            feature_maps
    ):
        objectness_outputs = []
        box_outputs = []

        for feature in feature_maps:
            x = F.relu(
                self.conv(feature),
                inplace=True
            )

            objectness = self.objectness(x)
            box_regression = self.box_regression(x)

            batch_size = feature.shape[0]

            objectness = objectness.permute(
                0,
                2,
                3,
                1
            ).contiguous().reshape(
                batch_size,
                -1
            )

            box_regression = box_regression.permute(
                0,
                2,
                3,
                1
            ).contiguous().reshape(
                batch_size,
                -1,
                4
            )

            objectness_outputs.append(
                objectness
            )

            box_outputs.append(
                box_regression
            )

        return (
            torch.cat(
                objectness_outputs,
                dim=1
            ),
            torch.cat(
                box_outputs,
                dim=1
            )
        )

def box_iou(
        boxes1: torch.Tensor,
        boxes2: torch.Tensor,
        eps: float = 1e-7
) -> torch.Tensor:
    if (
            boxes1.numel() == 0
            or boxes2.numel() == 0
    ):
        return torch.zeros(
            (
                boxes1.shape[0],
                boxes2.shape[0]
            ),
            device=boxes1.device
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

    area1 = (
            (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0)
            * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    )

    area2 = (
            (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0)
            * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    )

    union = (
            area1[:, None]
            + area2[None, :]
            - intersection
    )

    return (
            intersection
            / (union + eps)
    )


def encode_boxes(
        ground_truth: torch.Tensor,
        reference_boxes: torch.Tensor
) -> torch.Tensor:
    reference_widths = (
            reference_boxes[:, 2]
            - reference_boxes[:, 0]
    ).clamp(
        min=1e-6
    )

    reference_heights = (
            reference_boxes[:, 3]
            - reference_boxes[:, 1]
    ).clamp(
        min=1e-6
    )

    reference_center_x = (
            reference_boxes[:, 0]
            + 0.5 * reference_widths
    )

    reference_center_y = (
            reference_boxes[:, 1]
            + 0.5 * reference_heights
    )

    gt_widths = (
            ground_truth[:, 2]
            - ground_truth[:, 0]
    ).clamp(
        min=1e-6
    )

    gt_heights = (
            ground_truth[:, 3]
            - ground_truth[:, 1]
    ).clamp(
        min=1e-6
    )

    gt_center_x = (
            ground_truth[:, 0]
            + 0.5 * gt_widths
    )

    gt_center_y = (
            ground_truth[:, 1]
            + 0.5 * gt_heights
    )

    dx = (
            gt_center_x
            - reference_center_x
    ) / reference_widths

    dy = (
            gt_center_y
            - reference_center_y
    ) / reference_heights

    dw = torch.log(
        gt_widths
        / reference_widths
    )

    dh = torch.log(
        gt_heights
        / reference_heights
    )

    return torch.stack(
        [
            dx,
            dy,
            dw,
            dh
        ],
        dim=1
    )


def decode_boxes(
        deltas: torch.Tensor,
        reference_boxes: torch.Tensor
) -> torch.Tensor:
    widths = (
            reference_boxes[:, 2]
            - reference_boxes[:, 0]
    ).clamp(
        min=1e-6
    )

    heights = (
            reference_boxes[:, 3]
            - reference_boxes[:, 1]
    ).clamp(
        min=1e-6
    )

    center_x = (
            reference_boxes[:, 0]
            + 0.5 * widths
    )

    center_y = (
            reference_boxes[:, 1]
            + 0.5 * heights
    )

    dx = deltas[:, 0]
    dy = deltas[:, 1]

    dw = deltas[:, 2].clamp(
        min=-4.0,
        max=4.0
    )

    dh = deltas[:, 3].clamp(
        min=-4.0,
        max=4.0
    )

    predicted_center_x = (
            dx
            * widths
            + center_x
    )

    predicted_center_y = (
            dy
            * heights
            + center_y
    )

    predicted_widths = (
            torch.exp(dw)
            * widths
    )

    predicted_heights = (
            torch.exp(dh)
            * heights
    )

    x1 = (
            predicted_center_x
            - 0.5 * predicted_widths
    )

    y1 = (
            predicted_center_y
            - 0.5 * predicted_heights
    )

    x2 = (
            predicted_center_x
            + 0.5 * predicted_widths
    )

    y2 = (
            predicted_center_y
            + 0.5 * predicted_heights
    )

    return torch.stack(
        [
            x1,
            y1,
            x2,
            y2
        ],
        dim=1
    )


def clip_boxes(
        boxes: torch.Tensor,
        image_height: int,
        image_width: int
) -> torch.Tensor:
    boxes = boxes.clone()

    boxes[:, 0] = boxes[:, 0].clamp(
        min=0,
        max=image_width
    )

    boxes[:, 2] = boxes[:, 2].clamp(
        min=0,
        max=image_width
    )

    boxes[:, 1] = boxes[:, 1].clamp(
        min=0,
        max=image_height
    )

    boxes[:, 3] = boxes[:, 3].clamp(
        min=0,
        max=image_height
    )

    return boxes


def nms(
        boxes: torch.Tensor,
        scores: torch.Tensor,
        iou_threshold: float = 0.5
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
            boxes[current].unsqueeze(0),
            boxes[remaining]
        ).squeeze(0)

        order = remaining[
            ious <= iou_threshold
        ]

    return torch.stack(
        keep
    )

class ProposalGenerator:
    def __init__(
            self,
            pre_nms_top_n_train: int = 400,
            post_nms_top_n_train: int = 100,
            pre_nms_top_n_test: int = 200,
            post_nms_top_n_test: int = 50,
            nms_threshold: float = 0.7,
            min_size: float = 2.0
    ):
        self.pre_nms_top_n_train = pre_nms_top_n_train
        self.post_nms_top_n_train = post_nms_top_n_train
        self.pre_nms_top_n_test = pre_nms_top_n_test
        self.post_nms_top_n_test = post_nms_top_n_test
        self.nms_threshold = nms_threshold
        self.min_size = min_size

    def __call__(
            self,
            objectness_logits: torch.Tensor,
            box_deltas: torch.Tensor,
            anchors: torch.Tensor,
            image_height: int,
            image_width: int,
            training: bool
    ):
        proposals_per_image = []

        if training:
            pre_nms_top_n = self.pre_nms_top_n_train
            post_nms_top_n = self.post_nms_top_n_train
        else:
            pre_nms_top_n = self.pre_nms_top_n_test
            post_nms_top_n = self.post_nms_top_n_test

        for batch_index in range(
                objectness_logits.shape[0]
        ):
            scores = torch.sigmoid(
                objectness_logits[batch_index]
            )

            proposals = decode_boxes(
                box_deltas[batch_index],
                anchors
            )

            proposals = clip_boxes(
                proposals,
                image_height=image_height,
                image_width=image_width
            )

            widths = (
                    proposals[:, 2]
                    - proposals[:, 0]
            )

            heights = (
                    proposals[:, 3]
                    - proposals[:, 1]
            )

            valid = (
                    (widths >= self.min_size)
                    & (heights >= self.min_size)
            )

            proposals = proposals[valid]
            scores = scores[valid]

            if scores.numel() == 0:
                proposals_per_image.append(
                    anchors[:1].clone()
                )
                continue

            top_n = min(
                pre_nms_top_n,
                scores.numel()
            )

            top_indices = torch.topk(
                scores,
                k=top_n
            ).indices

            proposals = proposals[
                top_indices
            ]

            scores = scores[
                top_indices
            ]

            keep = nms(
                boxes=proposals,
                scores=scores,
                iou_threshold=self.nms_threshold
            )

            keep = keep[
                   :post_nms_top_n
                   ]

            proposals_per_image.append(
                proposals[keep]
            )

        return proposals_per_image

class MultiScaleRoIAlign(nn.Module):
    def __init__(self, output_size: int = 7):
        super().__init__()

        self.output_size = output_size

    def _choose_level(
            self,
            box: torch.Tensor
    ) -> int:
        width = (
                box[2]
                - box[0]
        ).clamp(
            min=1.0
        )

        height = (
                box[3]
                - box[1]
        ).clamp(
            min=1.0
        )

        size = torch.sqrt(
            width
            * height
        ).item()

        if size < 32:
            return 0

        if size < 64:
            return 1

        return 2

    def _pool_single(
            self,
            feature: torch.Tensor,
            box: torch.Tensor,
            image_height: int,
            image_width: int
    ) -> torch.Tensor:
        _, _, feature_height, feature_width = feature.shape

        scale_x = (
                feature_width
                / image_width
        )

        scale_y = (
                feature_height
                / image_height
        )

        x1 = box[0] * scale_x
        y1 = box[1] * scale_y
        x2 = box[2] * scale_x
        y2 = box[3] * scale_y

        sample_positions = (
                                   torch.arange(
                                       self.output_size,
                                       device=feature.device,
                                       dtype=feature.dtype
                                   )
                                   + 0.5
                           ) / self.output_size

        xs = (
                x1
                + sample_positions
                * (x2 - x1)
        ).clamp(
            min=0,
            max=max(feature_width - 1, 0)
        )

        ys = (
                y1
                + sample_positions
                * (y2 - y1)
        ).clamp(
            min=0,
            max=max(feature_height - 1, 0)
        )

        grid_y, grid_x = torch.meshgrid(
            ys,
            xs,
            indexing="ij"
        )

        if feature_width > 1:
            normalized_x = (
                    2.0
                    * grid_x
                    / (feature_width - 1)
                    - 1.0
            )
        else:
            normalized_x = torch.zeros_like(
                grid_x
            )

        if feature_height > 1:
            normalized_y = (
                    2.0
                    * grid_y
                    / (feature_height - 1)
                    - 1.0
            )
        else:
            normalized_y = torch.zeros_like(
                grid_y
            )

        grid = torch.stack(
            [
                normalized_x,
                normalized_y
            ],
            dim=-1
        ).unsqueeze(0)

        pooled = F.grid_sample(
            feature,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True
        )

        return pooled.squeeze(0)

    def forward(
            self,
            feature_maps,
            proposals_per_image,
            image_height: int,
            image_width: int
    ) -> torch.Tensor:
        pooled_features = []

        for batch_index, proposals in enumerate(
                proposals_per_image
        ):
            for proposal in proposals:
                level = self._choose_level(
                    proposal
                )

                feature = feature_maps[level][
                          batch_index:
                          batch_index + 1
                          ]

                pooled = self._pool_single(
                    feature=feature,
                    box=proposal,
                    image_height=image_height,
                    image_width=image_width
                )

                pooled_features.append(
                    pooled
                )

        if len(pooled_features) == 0:
            channels = feature_maps[0].shape[1]

            return torch.empty(
                (
                    0,
                    channels,
                    self.output_size,
                    self.output_size
                ),
                device=feature_maps[0].device
            )

        return torch.stack(
            pooled_features,
            dim=0
        )

class RoIHead(nn.Module):

    def __init__(
            self,
            channels: int = 128,
            roi_size: int = 7,
            hidden_dim: int = 256,
            num_classes_with_background: int = 4
    ):
        super().__init__()

        self.num_classes = num_classes_with_background

        flattened = (
                channels
                * roi_size
                * roi_size
        )

        self.fc1 = nn.Linear(
            flattened,
            hidden_dim
        )

        self.fc2 = nn.Linear(
            hidden_dim,
            hidden_dim
        )

        self.classifier = nn.Linear(
            hidden_dim,
            num_classes_with_background
        )

        self.box_regressor = nn.Linear(
            hidden_dim,
            (
                    num_classes_with_background
                    * 4
            )
        )

    def forward(
            self,
            pooled_features: torch.Tensor
    ):
        x = pooled_features.flatten(
            start_dim=1
        )

        x = F.relu(
            self.fc1(x),
            inplace=True
        )

        x = F.relu(
            self.fc2(x),
            inplace=True
        )

        class_logits = self.classifier(x)

        box_regression = self.box_regressor(x).view(
            x.shape[0],
            self.num_classes,
            4
        )

        return (
            class_logits,
            box_regression
        )

class FasterRCNNLoss:
    def __init__(
            self,
            rpn_positive_iou: float = 0.7,
            rpn_negative_iou: float = 0.3,
            rpn_batch_size: int = 256,
            rpn_positive_fraction: float = 0.5,
            roi_positive_iou: float = 0.5,
            roi_batch_size: int = 64,
            roi_positive_fraction: float = 0.25
    ):
        self.rpn_positive_iou = rpn_positive_iou
        self.rpn_negative_iou = rpn_negative_iou
        self.rpn_batch_size = rpn_batch_size
        self.rpn_positive_fraction = rpn_positive_fraction
        self.roi_positive_iou = roi_positive_iou
        self.roi_batch_size = roi_batch_size
        self.roi_positive_fraction = roi_positive_fraction

    def _subsample_labels(
            self,
            labels: torch.Tensor,
            batch_size: int,
            positive_fraction: float
    ) -> torch.Tensor:
        positive_indices = torch.where(
            labels == 1
        )[0]

        negative_indices = torch.where(
            labels == 0
        )[0]

        max_positive = int(
            batch_size
            * positive_fraction
        )

        num_positive = min(
            max_positive,
            positive_indices.numel()
        )

        num_negative = min(
            batch_size - num_positive,
            negative_indices.numel()
        )

        if positive_indices.numel() > 0:
            permutation = torch.randperm(
                positive_indices.numel(),
                device=labels.device
            )

            positive_indices = positive_indices[
                permutation[:num_positive]
            ]

        if negative_indices.numel() > 0:
            permutation = torch.randperm(
                negative_indices.numel(),
                device=labels.device
            )

            negative_indices = negative_indices[
                permutation[:num_negative]
            ]

        selected = torch.cat(
            [
                positive_indices,
                negative_indices
            ]
        )

        return selected

    def rpn_loss(
            self,
            objectness_logits: torch.Tensor,
            box_deltas: torch.Tensor,
            anchors: torch.Tensor,
            targets
    ):
        total_objectness = torch.tensor(
            0.0,
            device=objectness_logits.device
        )

        total_box = torch.tensor(
            0.0,
            device=objectness_logits.device
        )

        batch_size = objectness_logits.shape[0]

        for batch_index in range(
                batch_size
        ):
            gt_boxes = targets[batch_index][
                "boxes"
            ]

            ious = box_iou(
                anchors,
                gt_boxes
            )

            best_iou, best_gt_index = ious.max(
                dim=1
            )

            labels = torch.full(
                (anchors.shape[0],),
                fill_value=-1,
                dtype=torch.long,
                device=anchors.device
            )

            labels[
                best_iou < self.rpn_negative_iou
            ] = 0

            labels[
                best_iou >= self.rpn_positive_iou
            ] = 1

            best_anchor_for_gt = ious.argmax(
                dim=0
            )

            labels[
                best_anchor_for_gt
            ] = 1

            selected = self._subsample_labels(
                labels=labels,
                batch_size=self.rpn_batch_size,
                positive_fraction=self.rpn_positive_fraction
            )

            if selected.numel() > 0:
                objectness_target = labels[
                    selected
                ].float()

                total_objectness = (
                        total_objectness
                        + F.binary_cross_entropy_with_logits(
                            objectness_logits[
                                batch_index,
                                selected
                            ],
                            objectness_target
                        )
                )

            positive = torch.where(
                labels == 1
            )[0]

            if positive.numel() > 0:
                matched_gt = gt_boxes[
                    best_gt_index[positive]
                ]

                regression_targets = encode_boxes(
                    ground_truth=matched_gt,
                    reference_boxes=anchors[positive]
                )

                total_box = (
                        total_box
                        + F.smooth_l1_loss(
                            box_deltas[
                                batch_index,
                                positive
                            ],
                            regression_targets,
                            beta=1.0 / 9.0,
                            reduction="mean"
                        )
                )

        return (
            total_objectness / batch_size,
            total_box / batch_size
        )

    def prepare_roi_targets(
            self,
            proposals_per_image,
            targets
    ):
        sampled_proposals = []
        sampled_labels = []
        sampled_regression_targets = []

        for proposals, target in zip(
                proposals_per_image,
                targets
        ):
            gt_boxes = target[
                "boxes"
            ]

            gt_labels = target[
                "labels"
            ]

            proposals = torch.cat(
                [
                    proposals,
                    gt_boxes
                ],
                dim=0
            )

            ious = box_iou(
                proposals,
                gt_boxes
            )

            best_iou, best_gt_index = ious.max(
                dim=1
            )

            labels = torch.zeros(
                proposals.shape[0],
                dtype=torch.long,
                device=proposals.device
            )

            positive_mask = (
                    best_iou
                    >= self.roi_positive_iou
            )

            labels[
                positive_mask
            ] = gt_labels[
                best_gt_index[
                    positive_mask
                ]
            ]

            binary_labels = (
                    labels
                    > 0
            ).long()

            selected = self._subsample_labels(
                labels=binary_labels,
                batch_size=self.roi_batch_size,
                positive_fraction=self.roi_positive_fraction
            )

            proposals = proposals[
                selected
            ]

            labels = labels[
                selected
            ]

            matched_gt = gt_boxes[
                best_gt_index[
                    selected
                ]
            ]

            regression_targets = encode_boxes(
                ground_truth=matched_gt,
                reference_boxes=proposals
            )

            sampled_proposals.append(
                proposals
            )

            sampled_labels.append(
                labels
            )

            sampled_regression_targets.append(
                regression_targets
            )

        return (
            sampled_proposals,
            torch.cat(
                sampled_labels,
                dim=0
            ),
            torch.cat(
                sampled_regression_targets,
                dim=0
            )
        )

    def roi_loss(
            self,
            class_logits: torch.Tensor,
            box_regression: torch.Tensor,
            labels: torch.Tensor,
            regression_targets: torch.Tensor
    ):
        classification_loss = F.cross_entropy(
            class_logits,
            labels
        )

        positive = torch.where(
            labels > 0
        )[0]

        if positive.numel() == 0:
            box_loss = box_regression.sum() * 0.0
        else:
            predicted = box_regression[
                positive,
                labels[positive]
            ]

            box_loss = F.smooth_l1_loss(
                predicted,
                regression_targets[positive],
                beta=1.0,
                reduction="mean"
            )

        return (
            classification_loss,
            box_loss
        )

class FasterRCNN(nn.Module):
    def __init__(self, num_classes: int = 3):
        super().__init__()

        self.num_classes = num_classes
        self.num_classes_with_background = (
                num_classes
                + 1
        )

        self.backbone = FasterRCNNBackbone()

        self.fpn = FPN(
            in_channels=(128, 192, 256),
            out_channels=128
        )

        self.anchor_generator = AnchorGenerator(
            sizes=(16, 32, 64),
            aspect_ratios=(0.5, 1.0, 2.0)
        )

        self.rpn_head = RPNHead(
            channels=128,
            num_anchors=self.anchor_generator.num_anchors
        )

        self.proposal_generator = ProposalGenerator()

        self.roi_align = MultiScaleRoIAlign(
            output_size=7
        )

        self.roi_head = RoIHead(
            channels=128,
            roi_size=7,
            hidden_dim=256,
            num_classes_with_background=(
                self.num_classes_with_background
            )
        )

        self.loss_helper = FasterRCNNLoss()

    def _postprocess_detections(
            self,
            proposals_per_image,
            class_logits: torch.Tensor,
            box_regression: torch.Tensor,
            image_height: int,
            image_width: int,
            score_threshold: float = 0.05,
            nms_threshold: float = 0.5,
            detections_per_image: int = 50
    ):
        probabilities = F.softmax(
            class_logits,
            dim=1
        )

        predictions = []
        start = 0

        for proposals in proposals_per_image:
            count = proposals.shape[0]
            end = start + count

            image_probabilities = probabilities[
                                  start:end
                                  ]

            image_box_regression = box_regression[
                                   start:end
                                   ]

            all_boxes = []
            all_scores = []
            all_labels = []

            for class_index in range(
                    1,
                    self.num_classes_with_background
            ):
                scores = image_probabilities[
                         :,
                         class_index
                         ]

                keep_score = (
                        scores
                        >= score_threshold
                )

                if keep_score.sum() == 0:
                    continue

                class_scores = scores[
                    keep_score
                ]

                class_proposals = proposals[
                    keep_score
                ]

                class_deltas = image_box_regression[
                    keep_score,
                    class_index
                ]

                class_boxes = decode_boxes(
                    deltas=class_deltas,
                    reference_boxes=class_proposals
                )

                class_boxes = clip_boxes(
                    class_boxes,
                    image_height=image_height,
                    image_width=image_width
                )

                keep = nms(
                    boxes=class_boxes,
                    scores=class_scores,
                    iou_threshold=nms_threshold
                )

                all_boxes.append(
                    class_boxes[keep]
                )

                all_scores.append(
                    class_scores[keep]
                )

                all_labels.append(
                    torch.full(
                        (keep.numel(),),
                        fill_value=class_index,
                        dtype=torch.long,
                        device=class_scores.device
                    )
                )

            if len(all_boxes) == 0:
                predictions.append({
                    "boxes": torch.empty(
                        (0, 4),
                        device=class_logits.device
                    ),
                    "scores": torch.empty(
                        (0,),
                        device=class_logits.device
                    ),
                    "labels": torch.empty(
                        (0,),
                        dtype=torch.long,
                        device=class_logits.device
                    )
                })
            else:
                boxes = torch.cat(
                    all_boxes,
                    dim=0
                )

                scores = torch.cat(
                    all_scores,
                    dim=0
                )

                labels = torch.cat(
                    all_labels,
                    dim=0
                )

                order = torch.argsort(
                    scores,
                    descending=True
                )[:detections_per_image]

                predictions.append({
                    "boxes": boxes[order],
                    "scores": scores[order],
                    "labels": labels[order]
                })

            start = end

        return predictions

    def forward(
            self,
            images: torch.Tensor,
            targets=None
    ):
        image_height = images.shape[-2]
        image_width = images.shape[-1]

        backbone_features = self.backbone(
            images
        )

        feature_maps = self.fpn(
            backbone_features
        )

        anchors = self.anchor_generator.generate(
            feature_maps=feature_maps,
            image_height=image_height,
            image_width=image_width
        )

        objectness_logits, rpn_box_deltas = self.rpn_head(
            feature_maps
        )

        proposals_per_image = self.proposal_generator(
            objectness_logits=objectness_logits.detach(),
            box_deltas=rpn_box_deltas.detach(),
            anchors=anchors,
            image_height=image_height,
            image_width=image_width,
            training=(
                    targets is not None
            )
        )

        if targets is not None:
            rpn_objectness_loss, rpn_box_loss = (
                self.loss_helper.rpn_loss(
                    objectness_logits=objectness_logits,
                    box_deltas=rpn_box_deltas,
                    anchors=anchors,
                    targets=targets
                )
            )

            (
                sampled_proposals,
                roi_labels,
                roi_regression_targets
            ) = self.loss_helper.prepare_roi_targets(
                proposals_per_image=proposals_per_image,
                targets=targets
            )

            pooled_features = self.roi_align(
                feature_maps=feature_maps,
                proposals_per_image=sampled_proposals,
                image_height=image_height,
                image_width=image_width
            )

            class_logits, roi_box_regression = self.roi_head(
                pooled_features
            )

            roi_classification_loss, roi_box_loss = (
                self.loss_helper.roi_loss(
                    class_logits=class_logits,
                    box_regression=roi_box_regression,
                    labels=roi_labels,
                    regression_targets=roi_regression_targets
                )
            )

            return {
                "loss_rpn_objectness": rpn_objectness_loss,
                "loss_rpn_box_reg": rpn_box_loss,
                "loss_classifier": roi_classification_loss,
                "loss_box_reg": roi_box_loss
            }

        pooled_features = self.roi_align(
            feature_maps=feature_maps,
            proposals_per_image=proposals_per_image,
            image_height=image_height,
            image_width=image_width
        )

        class_logits, roi_box_regression = self.roi_head(
            pooled_features
        )

        return self._postprocess_detections(
            proposals_per_image=proposals_per_image,
            class_logits=class_logits,
            box_regression=roi_box_regression,
            image_height=image_height,
            image_width=image_width
        )

class SyntheticDetectionDataset(Dataset):

    def __init__(
            self,
            n_samples: int = 300,
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

        self.images = []
        self.targets = []

        for _ in range(n_samples):
            image = torch.randn(
                3,
                image_size,
                image_size,
                generator=generator
            ) * 0.04

            class_label = int(
                torch.randint(
                    low=1,
                    high=num_classes + 1,
                    size=(1,),
                    generator=generator
                ).item()
            )

            box_width = int(
                torch.randint(
                    low=image_size // 6,
                    high=image_size // 2,
                    size=(1,),
                    generator=generator
                ).item()
            )

            box_height = int(
                torch.randint(
                    low=image_size // 6,
                    high=image_size // 2,
                    size=(1,),
                    generator=generator
                ).item()
            )

            x1 = int(
                torch.randint(
                    low=0,
                    high=image_size - box_width,
                    size=(1,),
                    generator=generator
                ).item()
            )

            y1 = int(
                torch.randint(
                    low=0,
                    high=image_size - box_height,
                    size=(1,),
                    generator=generator
                ).item()
            )

            x2 = x1 + box_width
            y2 = y1 + box_height

            channel = (
                    class_label
                    - 1
            ) % 3

            image[
                channel,
                y1:y2,
                x1:x2
            ] += 1.0

            image[
                :,
                y1:y2,
                x1:x2
            ] += 0.15

            image = image.clamp(
                0,
                1
            )

            target = {
                "boxes": torch.tensor(
                    [[
                        float(x1),
                        float(y1),
                        float(x2),
                        float(y2)
                    ]],
                    dtype=torch.float32
                ),

                "labels": torch.tensor(
                    [class_label],
                    dtype=torch.long
                )
            }

            self.images.append(
                image
            )

            self.targets.append(
                target
            )

    def __len__(self):
        return self.n_samples

    def __getitem__(
            self,
            index
    ):
        return (
            self.images[index],
            {
                "boxes": self.targets[index][
                    "boxes"
                ].clone(),
                "labels": self.targets[index][
                    "labels"
                ].clone()
            }
        )


def collate_detection(
        batch
):
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

    return (
        images,
        targets
    )


def move_targets_to_device(
        targets,
        device
):
    result = []

    for target in targets:
        result.append({
            "boxes": target[
                "boxes"
            ].to(device),

            "labels": target[
                "labels"
            ].to(device)
        })

    return result


@torch.no_grad()
def detection_metrics(
        predictions,
        targets
):
    total_iou = 0.0
    total_class_correct = 0
    total_detection_correct = 0
    total = 0

    for prediction, target in zip(
            predictions,
            targets
    ):
        gt_box = target[
            "boxes"
        ][0]

        gt_label = int(
            target[
                "labels"
            ][0].item()
        )

        predicted_boxes = prediction[
            "boxes"
        ]

        predicted_labels = prediction[
            "labels"
        ]

        predicted_scores = prediction[
            "scores"
        ]

        if predicted_boxes.numel() == 0:
            total += 1
            continue

        ious = box_iou(
            predicted_boxes,
            gt_box.unsqueeze(0)
        ).squeeze(1)

        best_index = torch.argmax(
            ious
        )

        best_iou = float(
            ious[best_index].item()
        )

        predicted_label = int(
            predicted_labels[
                best_index
            ].item()
        )

        total_iou += best_iou

        if predicted_label == gt_label:
            total_class_correct += 1

        if (
                best_iou >= 0.5
                and predicted_label == gt_label
                and float(
                    predicted_scores[
                        best_index
                    ].item()
                ) >= 0.05
        ):
            total_detection_correct += 1

        total += 1

    if total == 0:
        return (
            0.0,
            0.0,
            0.0
        )

    return (
        total_iou / total,
        total_class_correct / total,
        total_detection_correct / total
    )


def train_epoch_custom(
        model: nn.Module,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device
):
    model.train()

    totals = {
        "loss": 0.0,
        "rpn_objectness": 0.0,
        "rpn_box": 0.0,
        "classifier": 0.0,
        "box_reg": 0.0
    }

    total_images = 0

    for images, targets in tqdm(
            dataloader,
            desc="Custom Faster R-CNN training"
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
            targets
        )

        loss = sum(
            loss_dict.values()
        )

        loss.backward()

        optimizer.step()

        batch_size = images.shape[0]

        totals["loss"] += (
                loss.item()
                * batch_size
        )

        totals["rpn_objectness"] += (
                loss_dict[
                    "loss_rpn_objectness"
                ].item()
                * batch_size
        )

        totals["rpn_box"] += (
                loss_dict[
                    "loss_rpn_box_reg"
                ].item()
                * batch_size
        )

        totals["classifier"] += (
                loss_dict[
                    "loss_classifier"
                ].item()
                * batch_size
        )

        totals["box_reg"] += (
                loss_dict[
                    "loss_box_reg"
                ].item()
                * batch_size
        )

        total_images += batch_size

    return {
        key: value / total_images
        for key, value in totals.items()
    }


@torch.no_grad()
def evaluate_custom(
        model: nn.Module,
        dataloader: DataLoader,
        device
):
    model.eval()

    all_predictions = []
    all_targets = []

    for images, targets in tqdm(
            dataloader,
            desc="Custom Faster R-CNN validation"
    ):
        images = images.to(
            device
        )

        device_targets = move_targets_to_device(
            targets,
            device
        )

        predictions = model(
            images
        )

        all_predictions.extend(
            predictions
        )

        all_targets.extend(
            device_targets
        )

    return detection_metrics(
        predictions=all_predictions,
        targets=all_targets
    )


def train_custom_model(
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
        losses = train_epoch_custom(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device
        )

        mean_iou, class_accuracy, detection_accuracy = evaluate_custom(
            model=model,
            dataloader=test_loader,
            device=device
        )

        print(
            f"Epoch {epoch + 1}/{epochs}"
        )

        print(
            f"Train total loss: {losses['loss']:.4f}"
        )

        print(
            f"RPN objectness:   {losses['rpn_objectness']:.4f}"
        )

        print(
            f"RPN box loss:     {losses['rpn_box']:.4f}"
        )

        print(
            f"ROI classifier:   {losses['classifier']:.4f}"
        )

        print(
            f"ROI box loss:     {losses['box_reg']:.4f}"
        )

        print(
            f"Validation mean IoU:      {mean_iou:.4f}"
        )

        print(
            f"Validation class acc:     {class_accuracy:.4f}"
        )

        print(
            f"Validation detection acc: {detection_accuracy:.4f}"
        )

        print()

    return evaluate_custom(
        model=model,
        dataloader=test_loader,
        device=device
    )


@torch.no_grad()
def inference_custom(
        model: nn.Module,
        image: torch.Tensor,
        device
):
    model.eval()

    image = image.unsqueeze(
        dim=0
    ).to(
        device
    )

    prediction = model(
        image
    )[0]

    return {
        "boxes": prediction[
            "boxes"
        ].cpu(),

        "scores": prediction[
            "scores"
        ].cpu(),

        "labels": prediction[
            "labels"
        ].cpu()
    }


def build_framework_model(
        num_classes_with_background: int = 4
):
    if fasterrcnn_resnet50_fpn is None:
        raise RuntimeError(
            "torchvision Faster R-CNN is unavailable"
        )

    return fasterrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=None,
        num_classes=num_classes_with_background,
        min_size=128,
        max_size=128,
        rpn_pre_nms_top_n_train=200,
        rpn_post_nms_top_n_train=100,
        rpn_pre_nms_top_n_test=100,
        rpn_post_nms_top_n_test=50,
        box_detections_per_img=50
    )


def train_epoch_framework(
        model: nn.Module,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device
):
    model.train()

    total_loss = 0.0
    total_images = 0

    for images, targets in tqdm(
            dataloader,
            desc="Torchvision Faster R-CNN training"
    ):
        image_list = [
            image.to(device)
            for image in images
        ]

        targets = move_targets_to_device(
            targets,
            device
        )

        optimizer.zero_grad()

        loss_dict = model(
            image_list,
            targets
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

        total_images += batch_size

    return (
            total_loss
            / total_images
    )


@torch.no_grad()
def evaluate_framework(
        model: nn.Module,
        dataloader: DataLoader,
        device
):
    model.eval()

    all_predictions = []
    all_targets = []

    for images, targets in tqdm(
            dataloader,
            desc="Torchvision Faster R-CNN validation"
    ):
        image_list = [
            image.to(device)
            for image in images
        ]

        predictions = model(
            image_list
        )

        predictions = [
            {
                "boxes": prediction[
                    "boxes"
                ],
                "scores": prediction[
                    "scores"
                ],
                "labels": prediction[
                    "labels"
                ]
            }
            for prediction in predictions
        ]

        targets = move_targets_to_device(
            targets,
            device
        )

        all_predictions.extend(
            predictions
        )

        all_targets.extend(
            targets
        )

    return detection_metrics(
        predictions=all_predictions,
        targets=all_targets
    )


def train_framework_model(
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        device,
        epochs: int = 1,
        lr: float = 1e-4
):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr
    )

    for epoch in range(epochs):
        loss = train_epoch_framework(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device
        )

        mean_iou, class_accuracy, detection_accuracy = evaluate_framework(
            model=model,
            dataloader=test_loader,
            device=device
        )

        print(
            f"Framework epoch {epoch + 1}/{epochs}"
        )

        print(
            f"Train loss: {loss:.4f}"
        )

        print(
            f"Validation mean IoU:      {mean_iou:.4f}"
        )

        print(
            f"Validation class acc:     {class_accuracy:.4f}"
        )

        print(
            f"Validation detection acc: {detection_accuracy:.4f}"
        )

        print()

    return evaluate_framework(
        model=model,
        dataloader=test_loader,
        device=device
    )


@torch.no_grad()
def inference_framework(
        model: nn.Module,
        image: torch.Tensor,
        device
):
    model.eval()

    prediction = model([
        image.to(device)
    ])[0]

    return {
        "boxes": prediction[
            "boxes"
        ].cpu(),

        "scores": prediction[
            "scores"
        ].cpu(),

        "labels": prediction[
            "labels"
        ].cpu()
    }


def count_parameters(
        model: nn.Module
) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def smoke_test_custom(
        device
):
    model = FasterRCNN(
        num_classes=3
    ).to(
        device
    )

    images = torch.rand(
        2,
        3,
        128,
        128,
        device=device
    )

    targets = [
        {
            "boxes": torch.tensor(
                [[20.0, 25.0, 65.0, 75.0]],
                device=device
            ),
            "labels": torch.tensor(
                [1],
                dtype=torch.long,
                device=device
            )
        },
        {
            "boxes": torch.tensor(
                [[50.0, 40.0, 100.0, 105.0]],
                device=device
            ),
            "labels": torch.tensor(
                [2],
                dtype=torch.long,
                device=device
            )
        }
    ]

    model.train()

    loss_dict = model(
        images,
        targets
    )

    loss = sum(
        loss_dict.values()
    )

    loss.backward()

    model.eval()

    with torch.no_grad():
        predictions = model(
            images
        )

    return (
        loss_dict,
        predictions
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
        batch_size=8,
        shuffle=True,
        collate_fn=collate_detection
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=collate_detection
    )

    image, true_target = test_dataset[0]

    print(
        "\n"
        "======================================="
    )

    print(
        "CUSTOM EDUCATIONAL FASTER R-CNN"
    )

    print(
        "======================================="
    )

    custom_model = FasterRCNN(
        num_classes=3
    ).to(
        device
    )

    print(
        custom_model
    )

    print(
        "Trainable parameters:",
        count_parameters(
            custom_model
        )
    )

    custom_metrics = train_custom_model(
        model=custom_model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        epochs=5,
        lr=1e-3
    )

    custom_prediction = inference_custom(
        model=custom_model,
        image=image,
        device=device
    )

    print(
        "\nCustom inference"
    )

    print(
        "True box:",
        true_target[
            "boxes"
        ]
    )

    print(
        "True label:",
        true_target[
            "labels"
        ]
    )

    print(
        "Predicted boxes:",
        custom_prediction[
            "boxes"
        ][:5]
    )

    print(
        "Predicted scores:",
        custom_prediction[
            "scores"
        ][:5]
    )

    print(
        "Predicted labels:",
        custom_prediction[
            "labels"
        ][:5]
    )

    framework_metrics = None

    if fasterrcnn_resnet50_fpn is not None:
        print(
            "\n"
            "======================================="
        )

        print(
            "TORCHVISION FASTER R-CNN RESNET50 FPN"
        )

        print(
            "======================================="
        )

        framework_model = build_framework_model(
            num_classes_with_background=4
        ).to(
            device
        )

        print(
            "Trainable parameters:",
            count_parameters(
                framework_model
            )
        )

        framework_metrics = train_framework_model(
            model=framework_model,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            epochs=1,
            lr=1e-4
        )

        framework_prediction = inference_framework(
            model=framework_model,
            image=image,
            device=device
        )

        print(
            "\nFramework inference"
        )

        print(
            "Predicted boxes:",
            framework_prediction[
                "boxes"
            ][:5]
        )

        print(
            "Predicted scores:",
            framework_prediction[
                "scores"
            ][:5]
        )

        print(
            "Predicted labels:",
            framework_prediction[
                "labels"
            ][:5]
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
        f"Custom mean IoU:      {custom_metrics[0]:.4f}"
    )

    print(
        f"Custom class acc:     {custom_metrics[1]:.4f}"
    )

    print(
        f"Custom detection acc: {custom_metrics[2]:.4f}"
    )

    if framework_metrics is not None:
        print()

        print(
            f"Framework mean IoU:      {framework_metrics[0]:.4f}"
        )

        print(
            f"Framework class acc:     {framework_metrics[1]:.4f}"
        )

        print(
            f"Framework detection acc: {framework_metrics[2]:.4f}"
        )


if __name__ == "__main__":
    main()
