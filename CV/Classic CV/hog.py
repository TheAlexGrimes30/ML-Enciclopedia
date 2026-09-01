import numpy as np

class HOGDescriptor:
    def __init__(
            self,
            cell_size: int = 8,
            block_size: int = 2,
            bins: int = 9
    ):
        self.cell_size = cell_size
        self.block_size = block_size
        self.bins = bins

        self.kernel_x = np.array([
            [-1, 0, 1]
        ], dtype=np.float32)

        self.kernel_y = np.array([
            [-1],
            [0],
            [1]
        ], dtype=np.float32)

    def convolve(self, image: np.ndarray, kernel: np.ndarray):
        h, w = image.shape

        kh, kw = kernel.shape

        pad_y = kh // 2
        pad_x = kw // 2

        padded = np.pad(
            image,
            (
                (pad_y, pad_y),
                (pad_x, pad_x)
            ),
            mode="edge"
        )

        output = np.zeros(
            (h, w),
            dtype=np.float32
        )

        for y in range(h):
            for x in range(w):
                patch = padded[
                        y:y + kh,
                        x:x + kw
                        ]

                output[y, x] = np.sum(
                    patch * kernel
                )

        return output

    def gradients(self, image):
        gx = self.convolve(
            image,
            self.kernel_x
        )

        gy = self.convolve(
            image,
            self.kernel_y
        )

        magnitude = np.sqrt(
            gx ** 2 + gy ** 2
        )

        angle = np.degrees(
            np.arctan2(
                gy,
                gx
            )
        )

        angle[angle < 0] += 180

        return magnitude, angle

    def cell_histogram(
            self,
            magnitude: np.ndarray,
            angle: np.ndarray
    ):
        h, w = magnitude.shape

        cells_y = h // self.cell_size
        cells_x = w // self.cell_size

        histograms = np.zeros(
            (
                cells_y,
                cells_x,
                self.bins
            ),
            dtype=np.float32
        )

        bin_width = 180 / self.bins

        for cy in range(cells_y):
            for cx in range(cells_x):

                y0 = cy * self.cell_size
                x0 = cx * self.cell_size

                mag_cell = magnitude[
                           y0:y0 + self.cell_size,
                           x0:x0 + self.cell_size
                           ]

                ang_cell = angle[
                           y0:y0 + self.cell_size,
                           x0:x0 + self.cell_size
                           ]

                for y in range(self.cell_size):
                    for x in range(self.cell_size):

                        mag = mag_cell[y, x]
                        ang = ang_cell[y, x]

                        bin_idx = int(
                            ang // bin_width
                        )

                        if bin_idx == self.bins:
                            bin_idx = 0

                        histograms[
                            cy,
                            cx,
                            bin_idx
                        ] += mag

        return histograms

    def normalize_blocks(
            self,
            histograms: np.ndarray
    ):
        cells_y, cells_x, _ = histograms.shape

        features = []

        for y in range(
                cells_y - self.block_size + 1
        ):
            for x in range(
                    cells_x - self.block_size + 1
            ):
                block = histograms[
                        y:y + self.block_size,
                        x:x + self.block_size
                        ]

                block = block.flatten()

                norm = np.sqrt(
                    np.sum(block ** 2)
                    + 1e-6
                )

                block = block / norm

                features.extend(
                    block
                )

        return np.array(
            features,
            dtype=np.float32
        )

    def apply(self, image: np.ndarray):
        image = image.astype(
            np.float32
        )

        magnitude, angle = self.gradients(
            image
        )

        histograms = self.cell_histogram(
            magnitude,
            angle
        )

        features = self.normalize_blocks(
            histograms
        )

        return features

image = np.array([
    [10, 10, 10, 10, 200, 200, 200, 200],
    [10, 10, 10, 10, 200, 200, 200, 200],
    [10, 10, 10, 10, 200, 200, 200, 200],
    [10, 10, 10, 10, 200, 200, 200, 200],
    [10, 10, 10, 10, 200, 200, 200, 200],
    [10, 10, 10, 10, 200, 200, 200, 200],
    [10, 10, 10, 10, 200, 200, 200, 200],
    [10, 10, 10, 10, 200, 200, 200, 200],
], dtype=np.uint8)


hog = HOGDescriptor(
    cell_size=4,
    block_size=2,
    bins=9
)

features_numpy = hog.apply(image)

print("HoG вручную:")
print(features_numpy)

print("\nРазмер:")
print(features_numpy.shape)
