import cv2
import numpy as np
from scipy.ndimage import gaussian_filter


class SIFTDetector:
    def __init__(
            self,
            octaves: int = 3,
            scales: int = 3,
            sigma: float = 1.6,
            threshold: float = 0.02
    ):
        self.octaves = octaves
        self.scales = scales
        self.sigma = sigma
        self.threshold = threshold

    def gaussian_pyramid(
            self,
            image: np.ndarray
    ):
        pyramid = []

        image = image.astype(
            np.float32
        ) / 255.0

        k = 2 ** (
                1 / self.scales
        )

        current = image

        for octave in range(
                self.octaves
        ):
            levels = []

            for scale in range(
                    self.scales + 3
            ):
                sigma = (
                        self.sigma
                        * k ** scale
                )

                blurred = gaussian_filter(
                    current,
                    sigma=sigma
                )

                levels.append(
                    blurred
                )

            pyramid.append(
                levels
            )

            current = levels[
                          self.scales
                      ][::2, ::2]

        return pyramid

    def dog_pyramid(
            self,
            gaussian_pyramid
    ):
        dog = []

        for octave in gaussian_pyramid:
            octave_dog = []

            for i in range(
                    len(octave) - 1
            ):
                diff = (
                        octave[i + 1]
                        - octave[i]
                )

                octave_dog.append(
                    diff
                )

            dog.append(
                octave_dog
            )

        return dog

    def is_extremum(
            self,
            previous: np.ndarray,
            current: np.ndarray,
            next_: np.ndarray,
            y: int,
            x: int
    ):
        value = current[
            y,
            x
        ]

        cube = np.stack([
            previous[
            y - 1:y + 2,
            x - 1:x + 2
            ],
            current[
            y - 1:y + 2,
            x - 1:x + 2
            ],
            next_[
            y - 1:y + 2,
            x - 1:x + 2
            ]
        ])

        neighbors = cube.flatten()

        neighbors = np.delete(
            neighbors,
            13
        )

        return (
                value > neighbors.max()
                or
                value < neighbors.min()
        )

    def detect(
            self,
            image: np.ndarray
    ):
        gaussian = self.gaussian_pyramid(
            image
        )

        dog = self.dog_pyramid(
            gaussian
        )

        keypoints = []

        for octave_idx, octave in enumerate(
                dog
        ):
            for scale in range(
                    1,
                    len(octave) - 1
            ):
                previous = octave[
                    scale - 1
                    ]

                current = octave[
                    scale
                ]

                next_ = octave[
                    scale + 1
                    ]

                h, w = current.shape

                for y in range(
                        1,
                        h - 1
                ):
                    for x in range(
                            1,
                            w - 1
                    ):
                        value = current[
                            y,
                            x
                        ]

                        if (
                                abs(value)
                                < self.threshold
                        ):
                            continue

                        if self.is_extremum(
                                previous,
                                current,
                                next_,
                                y,
                                x
                        ):
                            factor = (
                                    2 ** octave_idx
                            )

                            keypoints.append(
                                (
                                    x * factor,
                                    y * factor
                                )
                            )

        return keypoints

image = np.zeros(
    (128, 128),
    dtype=np.uint8
)

image[
    30:90,
    30:90
] = 255

image[
    50:70,
    50:70
] = 0


sift = SIFTDetector(
    octaves=3,
    scales=3,
    sigma=1.6,
    threshold=0.01
)

keypoints_numpy = sift.detect(
    image
)

print("SIFT вручную:")

for point in keypoints_numpy:
    print(point)

print(
    "\nКоличество:",
    len(keypoints_numpy)
)
