import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage.color import rgb2gray
import adrt

def calc_k(eps, output_shape0):
    
    return np.tan(np.linspace(-np.pi / 2 + eps, np.pi / 2 - eps, num=output_shape0))

def calc_b(input_image1, output_shape1):
    
    return np.linspace(-input_image1, input_image1, output_shape1)


def radon_transform(input_image, output_shape=None):
    """
    Discrete Radon Transform for lines
    :param input_image: image to be transformed
    :param output_shape: shape of the output image
    :return: result: transformed image
    """
    if input_image.ndim == 3:
        input_image = rgb2gray(input_image)

    if output_shape is None:
        output_shape = input_image.shape

    eps = 1.e-8
    k_values = calc_k(eps, output_shape[0])
    b_values = calc_b(input_image.shape[1],  output_shape[1])

    result = np.zeros(output_shape)
    for k_ind in range(len(k_values)):
        for b_ind in range(len(b_values)):
            for x in np.arange(input_image.shape[0]):
                y = int(np.round(x * k_values[k_ind] + b_values[b_ind]))
                if 0 <= y < input_image.shape[1]:
                    result[k_ind][b_ind] += input_image[x][y]
    result = result * 255 / np.max(result)

    return result


#plt.title("Image")
#image = np.diag(np.ones(128))
#plt.imshow(image)
#plt.show()

#transformed_image = radon_transform(image)
#cv2.imwrite('tranformed_img.jpg', transformed_image)

#transformed_image = adrt.adrt(image)
#transformed_image = adrt.utils.stitch_adrt(transformed_image)
#plt.title("Image")
#plt.imshow(transformed_image)
#plt.show()