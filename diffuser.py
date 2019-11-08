import numpy as np
from numba import njit


@njit
def _calculate_diffusivity_matrix(img_with_border, border_width, variance):
    diffus_matrix = np.zeros(img_with_border.shape)
    for i in range(border_width, diffus_matrix.shape[0] - border_width):
        for j in range(border_width, diffus_matrix.shape[1] - border_width):
            pm1_x = img_with_border[i, j - 1]
            pp1_x = img_with_border[i, j + 1]
            derivative_x = (pp1_x - pm1_x) / float(2)
            pm1_y = img_with_border[i - 1, j]
            pp1_y = img_with_border[i + 1, j]
            derivative_y = (pp1_y - pm1_y) / float(2)
            grad_square = (derivative_x ** 2) + (derivative_y ** 2)
            diffus_matrix[i, j] = np.exp(-grad_square / variance)
    return diffus_matrix


@njit
def _diffuse(img_with_border, diffus_matrix, border_width, tau):
    assert img_with_border.shape == diffus_matrix.shape
    final_image = np.zeros(img_with_border.shape)
    diffus_mid = np.zeros((*diffus_matrix.shape, 4))
    for i in range(border_width, img_with_border.shape[0] - border_width):
        for j in range(border_width, img_with_border.shape[1] - border_width):
            # --- top ---
            diffus_mid[i, j, 0] = (diffus_matrix[i, j] + diffus_matrix[i - 1, j]) / 2
            # --- right ---
            diffus_mid[i, j, 1] = (diffus_matrix[i, j] + diffus_matrix[i, j + 1]) / 2
            # --- bottom ---
            diffus_mid[i, j, 2] = (diffus_matrix[i, j] + diffus_matrix[i + 1, j]) / 2
            # --- left ---
            diffus_mid[i, j, 3] = (diffus_matrix[i, j] + diffus_matrix[i, j - 1]) / 2
            outgoing = (1 - tau * np.sum(diffus_mid[i, j])) * img_with_border[i, j]
            incoming = tau * (diffus_mid[i, j, 0] * img_with_border[i - 1, j] +
                              diffus_mid[i, j, 1] * img_with_border[i, j + 1] +
                              diffus_mid[i, j, 2] * img_with_border[i + 1, j] +
                              diffus_mid[i, j, 3] * img_with_border[i, j - 1])
            final_image[i, j] = outgoing + incoming
    return final_image


def nonlinear_isotropic_diffusion(img_with_border, border_width):
    steps = 100
    variance = 40
    tau = 0.05
    for i in range(steps):
        # --- diffusivity matrix ---
        diffus_matrix = _calculate_diffusivity_matrix(img_with_border, border_width, variance)
        # --- diffuse ---
        img_with_border = _diffuse(img_with_border, diffus_matrix, border_width, tau)
    return img_with_border
