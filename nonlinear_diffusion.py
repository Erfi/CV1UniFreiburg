from numba import njit
import cv2
import numpy as np

from loader import Loader


@njit
def calculate_diffusivity(img_obj, org_img_shape, variance):
    x, y = org_img_shape
    diffusivity_matrix = np.zeros((x+2, y+2))
    # hard-coding ahead :(
    for i in range(1, x + 3):
        for j in range(1, y + 3):
            x_derivative = (img_obj[i+1, j] - img_obj[i-1, j])/2
            y_derivative = (img_obj[i, j+1] - img_obj[i, j-1])/2
            grad_square = x_derivative ** 2 + y_derivative ** 2
            diffusivity_matrix[i-1][j-1] = np.e ** - (grad_square / variance)
    # print(diffusivity_matrix.shape)
    return diffusivity_matrix


@njit
def diffuse(img_obj, diffusivity_matrix, tau, org_img_shape):
    x, y = org_img_shape
    new_img_obj = np.zeros(org_img_shape)
    # hard-coding ahead :(
    # index assignments are really crazy.
    for i in range(1, x+1):
        for j in range(1, y+1):
            diffusivity_i_half_pos = (diffusivity_matrix[i + 1][j] + diffusivity_matrix[i][j]) / 2
            diffusivity_i_half_neg = (diffusivity_matrix[i - 1][j] + diffusivity_matrix[i][j]) / 2
            diffusivity_j_half_pos = (diffusivity_matrix[i][j + 1] + diffusivity_matrix[i][j]) / 2
            diffusivity_j_half_neg = (diffusivity_matrix[i][j - 1] + diffusivity_matrix[i][j]) / 2
            central_pixel_contrib = (1 - tau * (diffusivity_i_half_pos + diffusivity_i_half_neg + diffusivity_j_half_pos + diffusivity_j_half_neg)) * img_obj[i+1][j+1]
            neighbour_pixel_contrib = tau * (diffusivity_i_half_pos * img_obj[i+2][j+1] +
                                             diffusivity_i_half_neg * img_obj[i][j+1] +
                                             diffusivity_j_half_pos * img_obj[i+1][j+2] +
                                             diffusivity_j_half_neg * img_obj[i+1][j])
            new_img_obj[i - 1][j - 1] = central_pixel_contrib + neighbour_pixel_contrib
    return new_img_obj


# @njit
def nonlinear_isotropic_diffusion(img_obj, org_img_shape):
    steps = 30
    variance = steps * 2
    variance = 50
    tau = 0.025
    for i in range(steps):
        diffusivity_matrix = calculate_diffusivity(img_obj, org_img_shape, variance)
        new_img_obj = diffuse(img_obj, diffusivity_matrix, tau, org_img_shape)
        img_obj = Loader.load_image_with_mirrored_border(new_img_obj, 2)
    return new_img_obj


img_obj = cv2.imread("data/BoatsNoise10.pgm", cv2.IMREAD_GRAYSCALE)
org_img_shape = img_obj.shape
img_obj = Loader.load_image_with_mirrored_border(img_obj, 2)
# print(img_obj.shape)
img_obj = nonlinear_isotropic_diffusion(img_obj, org_img_shape)
cv2.imwrite("data/BoatsProcessed.jpeg", img_obj)