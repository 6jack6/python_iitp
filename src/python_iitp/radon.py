from typing import Any

import cv2
from matplotlib import pyplot as plt
import numpy as np
from skimage.color import rgb2gray


def radon_transform(input_image: np.ndarray, output_shape: Any = None) -> np.ndarray:
    """Discrete Radon Transform for lines.

    Args:
        input_image: Image to be transformed.
        output_shape: Shape of the output image (None).

    Returns:
        Transformed image.

    Example:
        >>> from PIL import Image
        >>> img = Image.open('path/to/image.jpg')
        >>> transformed_image = radon_transform(image)
        True
    """
    if input_image.ndim == 3:
        input_image = rgb2gray(input_image)

    if output_shape is None:
        output_shape = input_image.shape

    eps = 1.0e-8
    k_values = np.tan(
        np.linspace(-np.pi / 2 + eps, np.pi / 2 - eps, num=output_shape[0])
    )
    b_values = np.linspace(-input_image.shape[1], input_image.shape[1], output_shape[1])

    result = np.zeros(output_shape)
    for k_ind in range(len(k_values)):
        for b_ind in range(len(b_values)):
            for x in np.arange(input_image.shape[0]):
                y = int(np.round(x * k_values[k_ind] + b_values[b_ind]))
                if 0 <= y < input_image.shape[1]:
                    result[k_ind][b_ind] += input_image[x][y]
    result = result * 255 / np.max(result)

    return result


plt.title("Image")
image = np.diag(np.ones(100))
plt.imshow(image)
plt.show()

transformed_image = radon_transform(image)
cv2.imwrite("tranformed_img.jpg", transformed_image)

plt.title("Image")
plt.imshow(transformed_image)
plt.show()
