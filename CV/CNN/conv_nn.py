import numpy as np
import torch
from torch import nn

class Conv2D:
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
        seed: int = 42,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        rng = np.random.default_rng(seed)

        scale = np.sqrt(
            2.0 / (
                in_channels
                * kernel_size
                * kernel_size
            )
        )

        self.weight = (
            rng.standard_normal(
                (
                    out_channels,
                    in_channels,
                    kernel_size,
                    kernel_size,
                )
            )
            * scale
        )

        self.bias = (
            np.zeros(out_channels)
            if bias
            else None
        )

        self.x = None
        self.x_padded = None

    def forward(self, x):
        n, _, h, w = x.shape

        out_h = (
            h
            + 2 * self.padding
            - self.kernel_size
        ) // self.stride + 1

        out_w = (
            w
            + 2 * self.padding
            - self.kernel_size
        ) // self.stride + 1

        x_padded = np.pad(
            x,
            (
                (0, 0),
                (0, 0),
                (self.padding, self.padding),
                (self.padding, self.padding),
            ),
        )

        out = np.zeros(
            (
                n,
                self.out_channels,
                out_h,
                out_w,
            )
        )

        for batch in range(n):
            for out_channel in range(
                self.out_channels
            ):
                for i in range(out_h):
                    for j in range(out_w):
                        y = i * self.stride
                        x_pos = j * self.stride

                        patch = x_padded[
                            batch,
                            :,
                            y:y + self.kernel_size,
                            x_pos:x_pos + self.kernel_size,
                        ]

                        out[
                            batch,
                            out_channel,
                            i,
                            j,
                        ] = np.sum(
                            patch
                            * self.weight[out_channel]
                        )

                        if self.bias is not None:
                            out[
                                batch,
                                out_channel,
                                i,
                                j,
                            ] += self.bias[out_channel]

        self.x = x
        self.x_padded = x_padded

        return out

    def backward(self, grad_output):
        n, _, h, w = self.x.shape

        _, _, out_h, out_w = (
            grad_output.shape
        )

        grad_x_padded = np.zeros_like(
            self.x_padded
        )

        self.grad_weight = np.zeros_like(
            self.weight
        )

        self.grad_bias = (
            np.zeros_like(self.bias)
            if self.bias is not None
            else None
        )

        for batch in range(n):
            for out_channel in range(
                self.out_channels
            ):
                for i in range(out_h):
                    for j in range(out_w):
                        y = i * self.stride
                        x_pos = j * self.stride

                        grad = grad_output[
                            batch,
                            out_channel,
                            i,
                            j,
                        ]

                        patch = self.x_padded[
                            batch,
                            :,
                            y:y + self.kernel_size,
                            x_pos:x_pos + self.kernel_size,
                        ]

                        self.grad_weight[
                            out_channel
                        ] += patch * grad

                        grad_x_padded[
                            batch,
                            :,
                            y:y + self.kernel_size,
                            x_pos:x_pos + self.kernel_size,
                        ] += (
                            self.weight[out_channel]
                            * grad
                        )

                        if self.grad_bias is not None:
                            self.grad_bias[
                                out_channel
                            ] += grad

        if self.padding == 0:
            return grad_x_padded

        return grad_x_padded[
            :,
            :,
            self.padding:self.padding + h,
            self.padding:self.padding + w,
        ]


def mse(pred, target):
    diff = pred - target

    loss = np.mean(diff ** 2)

    grad = (
        2 * diff
        / diff.size
    )

    return loss, grad


if __name__ == "__main__":
    rng = np.random.default_rng(1)

    x = rng.normal(
        size=(2, 3, 6, 6)
    )

    conv = Conv2D(
        in_channels=3,
        out_channels=4,
        kernel_size=3,
        stride=1,
        padding=1,
    )

    y = conv.forward(x)

    target = rng.normal(
        size=y.shape
    )

    loss, grad_output = mse(
        y,
        target,
    )

    grad_x = conv.backward(
        grad_output
    )

    print("NumPy")
    print("output:", y.shape)
    print("loss:", loss)
    print("dX:", grad_x.shape)
    print("dW:", conv.grad_weight.shape)
    print("db:", conv.grad_bias.shape)

    try:

        torch.set_default_dtype(
            torch.float64
        )

        torch_conv = nn.Conv2d(
            in_channels=3,
            out_channels=4,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        with torch.no_grad():
            torch_conv.weight.copy_(
                torch.from_numpy(
                    conv.weight
                )
            )

            torch_conv.bias.copy_(
                torch.from_numpy(
                    conv.bias
                )
            )

        x_torch = torch.tensor(
            x,
            requires_grad=True,
        )

        target_torch = torch.tensor(
            target
        )

        y_torch = torch_conv(
            x_torch
        )

        loss_torch = torch.mean(
            (
                y_torch
                - target_torch
            ) ** 2
        )

        loss_torch.backward()

        print("\nNumPy vs PyTorch")

        print(
            "output:",
            np.max(
                np.abs(
                    y
                    - y_torch
                    .detach()
                    .numpy()
                )
            ),
        )

        print(
            "dX:",
            np.max(
                np.abs(
                    grad_x
                    - x_torch.grad.numpy()
                )
            ),
        )

        print(
            "dW:",
            np.max(
                np.abs(
                    conv.grad_weight
                    - torch_conv
                    .weight
                    .grad
                    .numpy()
                )
            ),
        )

        print(
            "db:",
            np.max(
                np.abs(
                    conv.grad_bias
                    - torch_conv
                    .bias
                    .grad
                    .numpy()
                )
            ),
        )

    except ImportError:
        print(
            "\nPyTorch не установлен"
        )