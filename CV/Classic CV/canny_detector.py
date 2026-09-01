import cv2
import numpy as np


class CannyDetector:
    def __init__(
            self,
            low_threshold: int = 50,
            high_threshold: int = 100
    ):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

        self.kernel_x = np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=np.float32)

        self.kernel_y = np.array([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=np.float32)

    def convolve(self, image: np.ndarray, kernel: np.ndarray):
        h, w = image.shape

        padded = np.pad(
            image,
            1,
            mode="edge"
        )

        output = np.zeros(
            (h, w),
            dtype=np.float32
        )

        for y in range(h):
            for x in range(w):
                patch = padded[
                        y:y + 3,
                        x:x + 3
                        ]

                output[y, x] = np.sum(
                    patch * kernel
                )

        return output

    def non_max_suppression(self, magnitude: np.ndarray, angle: np.ndarray):
        h, w = magnitude.shape

        output = np.zeros_like(
            magnitude
        )

        angle = (
            angle * 180 / np.pi
        )

        angle[angle < 0] += 180

        for y in range(1, h - 1):
            for x in range(1, w - 1):

                current = magnitude[y, x]
                direction = angle[y, x]

                if (
                        direction < 22.5
                        or direction >= 157.5
                ):
                    n1 = magnitude[y, x - 1]
                    n2 = magnitude[y, x + 1]

                elif direction < 67.5:
                    n1 = magnitude[y - 1, x + 1]
                    n2 = magnitude[y + 1, x - 1]

                elif direction < 112.5:
                    n1 = magnitude[y - 1, x]
                    n2 = magnitude[y + 1, x]

                else:
                    n1 = magnitude[y - 1, x - 1]
                    n2 = magnitude[y + 1, x + 1]

                if (
                        current >= n1
                        and current >= n2
                ):
                    output[y, x] = current

        return output

    def threshold(self, image):
        result = np.zeros_like(
            image,
            dtype=np.uint8
        )

        strong = image >= self.high_threshold

        weak = (
                (image >= self.low_threshold)
                & (image < self.high_threshold)
        )

        result[strong] = 255
        result[weak] = 75

        return result

    def hysteresis(self, image):
        h, w = image.shape

        for y in range(1, h - 1):
            for x in range(1, w - 1):

                if image[y, x] == 75:

                    patch = image[
                            y - 1:y + 2,
                            x - 1:x + 2
                            ]

                    if np.any(
                            patch == 255
                    ):
                        image[y, x] = 255
                    else:
                        image[y, x] = 0

        return image

    def apply(self, image: np.ndarray):
        image = image.astype(
            np.float32
        )

        image = cv2.GaussianBlur(
            image,
            (3, 3),
            1
        )

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

        angle = np.arctan2(
            gy,
            gx
        )

        if magnitude.max() > 0:
            magnitude = (
                    magnitude
                    / magnitude.max()
                    * 255
            )

        thin_edges = self.non_max_suppression(
            magnitude,
            angle
        )

        thresholded = self.threshold(
            thin_edges
        )

        edges = self.hysteresis(
            thresholded
        )

        return edges

    def apply_cv2(self, image):
        return cv2.Canny(
            image,
            self.low_threshold,
            self.high_threshold
        )

image = np.array([
    [10, 10, 10, 200, 200],
    [10, 10, 10, 200, 200],
    [10, 10, 10, 200, 200],
    [10, 10, 10, 200, 200],
    [10, 10, 10, 200, 200],
], dtype=np.uint8)


canny = CannyDetector(
    low_threshold=50,
    high_threshold=100
)

edges_numpy = canny.apply(image)
edges_cv2 = canny.apply_cv2(image)

print("Canny вручную:")
print(edges_numpy)

print("\nCanny OpenCV:")
print(edges_cv2)