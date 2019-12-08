"""
Various image filters to apply to images in order to have smooth result

Author: Mehran Ahkami <ahkami.mehran@gmail.com

Good source of Gaussina Filter:
https://homepages.inf.ed.ac.uk/rbf/HIPR2/gsmooth.htm

Diffusion Filter:
https://pastebin.com/sBsPX4Y7
"""

import sys
from loader import ImageHandler
import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from numba import njit


class ImageFilter:
    def __init__(self, image, load_type=cv2.IMREAD_GRAYSCALE):
        self.img_handler = ImageHandler()
        self.image = self.img_handler.load_image(image, load_type)

    def get_origshape(self):
        return self.img_handler.get_originalShape()

    def first_derivative2D_v1(self):
        image_shape = self.image.shape
        first_driv = np.copy(self.image)
        for i in range(1, image_shape[0] - 1, 1):
            for j in range(1, image_shape[1] - 1, 1):
                first_driv[i, j] = (self.image[i - 1, j] - self.image[i + 1, j]) + (
                    self.image[i, j - 1] - self.image[i, j + 1]
                )

        return first_driv

    def second_derivative2D_v1(self):
        f_d = self.first_derivative2D_v1()
        shape = f_d.shape
        second_driv = np.copy(f_d)
        for i in range(1, shape[0] - 1, 1):
            for j in range(1, shape[1] - 1, 1):
                second_driv[i, j] = (f_d[i - 1, j] - f_d[i + 1, j]) + (
                    f_d[i, j - 1] - f_d[i, j + 1]
                )
        return second_driv

    def discrete_gradient(self, image, axis):
        image_shape = image.shape
        first_grad = np.copy(image)
        if axis == "x":  # grad with respect to x
            for y in range(1, image_shape[0] - 1, 1):
                for x in range(1, image_shape[1] - 1, 1):
                    first_grad[y, x] = (image[y, x - 1] - image[y, x + 1]) / float(2)
        else:  # grad with respct to y
            for y in range(1, image_shape[0] - 1, 1):
                for x in range(1, image_shape[1] - 1, 1):
                    first_grad[y, x] = (image[y - 1, x] - image[y + 1, x]) / float(2)
        return first_grad

    def diffusition_matrix(self, image, variance):
        if variance == 0:
            raise EnvironmentError
        diff_mat = np.copy(image)
        dx = self.discrete_gradient(image, "x")
        dy = self.discrete_gradient(image, "y")
        diff_mat = np.power(dx, 2) + np.power(dy, 2)
        return np.exp(-diff_mat / np.power(variance, 2))

    def nonlinear_diffusion(self, image, tau, variance):
        image_shape = image.shape
        output = np.copy(image)
        dm = self.diffusition_matrix(image, variance)
        for y in range(1, image_shape[0] - 1, 1):
            for x in range(1, image_shape[1] - 1, 1):
                output[y, x] = (
                    1
                    - tau
                    * (
                        ((dm[y, x + 1] + dm[y, x]) / 2)
                        + ((dm[y, x - 1] + dm[y, x]) / 2)
                        + ((dm[y + 1, x] + dm[y, x]) / 2)
                        + ((dm[y - 1, x] + dm[y, x]) / 2)
                    )
                ) * image[y, x] + tau * (
                    ((dm[y, x + 1] + dm[y, x]) / 2) * image[y, x + 1]
                    + ((dm[y, x - 1] + dm[y, x]) / 2) * image[y, x - 1]
                    + ((dm[y + 1, x] + dm[y, x]) / 2) * image[y + 1, x]
                    + ((dm[y - 1, x] + dm[y, x]) / 2) * image[y - 1, x]
                )
        return output


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("give file name, tau, variance, steps, save path")
        sys.exit()
    image_name = sys.argv[1]
    tau = float(sys.argv[2])
    variance = float(sys.argv[3])
    steps = int(sys.argv[4])
    save_path = sys.argv[5]
    f = ImageFilter(image_name)
    image = f.image
    for i in range(steps):
        image = f.nonlinear_diffusion(image, tau, variance)
    np.delete(image, 0, 0)
    np.delete(image, -1, 0)
    np.delete(image, 0, 1)
    np.delete(image, -1, 1)
    handler = ImageHandler()
    handler.save_image(image, name=save_path)
