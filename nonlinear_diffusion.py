from numba import njit, cuda
import cv2
import numpy as np

from loader import Loader


@njit(parallel=True, nogil=True)
def calculate_diffusivity(img_obj, shape, variance):
    x, y = shape
    diffusivity_matrix = np.zeros((x+2, y+2))
    # hard-coding ahead :(
    for i in range(x + 1):
        for j in range(y + 1):
            x_derivative = (img_obj[i+1, j] - img_obj[i-1, j])/2
            y_derivative = (img_obj[i, j+1] - img_obj[i, j-1])/2
            grad_square = x_derivative ** 2 + y_derivative ** 2
            diffusivity_matrix[i][j] = np.e ** - (grad_square / variance)

    # print(diffusivity_matrix.shape)
    return diffusivity_matrix


@njit(parallel=True, nogil=True)
def diffuse(img_obj, diffusivity_matrix, tau):
    x, y = diffusivity_matrix.shape
    new_img_obj = np.zeros(diffusivity_matrix.shape)
    # hard-coding ahead :(
    for i in range(x - 1):
        for j in range(y - 1):
            diffusivity_i_half_pos = (diffusivity_matrix[i + 1][j] + diffusivity_matrix[i][j]) / 2
            diffusivity_i_half_neg = (diffusivity_matrix[i - 1][j] + diffusivity_matrix[i][j]) / 2
            diffusivity_j_half_pos = (diffusivity_matrix[i][j + 1] + diffusivity_matrix[i][j]) / 2
            diffusivity_j_half_neg = (diffusivity_matrix[i][j - 1] + diffusivity_matrix[i][j]) / 2
            central_pixel_contrib = (1 - tau * (diffusivity_i_half_pos + diffusivity_i_half_neg + diffusivity_j_half_pos + diffusivity_j_half_neg)) * img_obj[i][j]
            neighbour_pixel_contrib = tau * (diffusivity_i_half_pos * img_obj[i+1][j] +
                                             diffusivity_i_half_neg * img_obj[i-1][j] +
                                             diffusivity_j_half_pos * img_obj[i][j+1] +
                                             diffusivity_j_half_neg * img_obj[i][j-1])
            new_img_obj[i, j] = central_pixel_contrib + neighbour_pixel_contrib
    return new_img_obj


def nonlinear_isotropic_diffusion(img_obj, shape):
    steps = 100
    variance = steps * 2
    tau = 0.05
    for i in range(steps):
        diffusivity_matrix = calculate_diffusivity(img_obj, shape, 50)
        new_img_obj = diffuse(img_obj, diffusivity_matrix, tau)
        # i dont know why i did it.
        img_obj = new_img_obj
    return img_obj


img_obj = cv2.imread("data/BoatsNoise10.pgm", cv2.IMREAD_GRAYSCALE)
shape = img_obj.shape
img_obj = Loader.load_image_with_mirrored_border(img_obj, 2)
# print(img_obj.shape)
img_obj = nonlinear_isotropic_diffusion(img_obj, shape)
cv2.imwrite("data/BoatsProcessed.jpeg", img_obj)