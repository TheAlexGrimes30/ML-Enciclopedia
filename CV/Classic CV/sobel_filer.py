import cv2
import numpy as np


class SobelFilter:
    def __init__(self):
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

    def apply(self, image):
        image = image.astype(np.float32)

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

        magnitude = (
                magnitude
                / magnitude.max()
                * 255
        )

        return magnitude.astype(np.uint8)

    def apply_cv2(self, image: np.ndarray):
        """
        Реализация Sobel через OpenCV.
        """

        gx = cv2.Sobel(
            image,
            cv2.CV_32F,
            dx=1,
            dy=0,
            ksize=3
        )

        gy = cv2.Sobel(
            image,
            cv2.CV_32F,
            dx=0,
            dy=1,
            ksize=3
        )

        magnitude = cv2.magnitude(
            gx,
            gy
        )

        if magnitude.max() > 0:
            magnitude = (
                    magnitude
                    / magnitude.max()
                    * 255
            )

        return magnitude.astype(np.uint8)


image = np.array([
    [10, 10, 10, 200, 200],
    [10, 10, 10, 200, 200],
    [10, 10, 10, 200, 200],
    [10, 10, 10, 200, 200],
    [10, 10, 10, 200, 200],
], dtype=np.uint8)


sobel = SobelFilter()

edges_numpy = sobel.apply(image)
edges_cv2 = sobel.apply_cv2(image)

print("Исходные данные:")
print(image)

print("\nSobel вручную:")
print(edges_numpy)

print("\nSobel через OpenCV:")
print(edges_cv2)