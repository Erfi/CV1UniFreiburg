import numpy as np
from loader import Loader
from numba import njit


@njit
def _calculate_diffusivity_matrix(img_with_border, border_width, variance):
    gauss = np.zeros(img_with_border.shape)
    for i in range(border_width, gauss.shape[0] - border_width):
        for j in range(border_width, gauss.shape[1] - border_width):
            pm1_x = img_with_border[i, j - 1]
            pp1_x = img_with_border[i, j + 1]
            derivative_x = (pp1_x - pm1_x) / float(2)
            pm1_y = img_with_border[i - 1, j]
            pp1_y = img_with_border[i + 1, j]
            derivative_y = (pp1_y - pm1_y) / float(2)
            grad_square = (derivative_x ** 2) + (derivative_y ** 2)
            gauss[i, j] = np.exp(-grad_square / variance)
    return gauss


@njit
def _calculate_in_between_diffusivity(gauss, border_width):
    g_middle = np.zeros((*gauss.shape, 4))
    for i in range(border_width, gauss.shape[0] - border_width):
        for j in range(border_width, gauss.shape[1] - border_width):
            # --- top ---
            g_middle[i, j, 0] = (gauss[i, j] + gauss[i - 1, j]) / 2
            # --- right ---
            g_middle[i, j, 1] = (gauss[i, j] + gauss[i, j + 1]) / 2
            # --- bottom ---
            g_middle[i, j, 2] = (gauss[i, j] + gauss[i + 1, j]) / 2
            # --- left ---
            g_middle[i, j, 3] = (gauss[i, j] + gauss[i, j - 1]) / 2

    return g_middle


@njit
def _diffuse(img_with_border, g_middle, border_width, tau):
    img_shape = (img_with_border.shape[0] - 2 * border_width, img_with_border.shape[1] - 2 * border_width)
    final_image = np.zeros(img_shape)
    for i in range(border_width, img_with_border.shape[0] - border_width):
        for j in range(border_width, img_with_border.shape[1] - border_width):
            outgoing = ((1 - tau) * np.sum(g_middle[i, j])) * img_with_border[i, j]
            incoming = tau * (g_middle[i, j, 0] * img_with_border[i - 1, j] +
                              g_middle[i, j, 1] * img_with_border[i, j + 1] +
                              g_middle[i, j, 2] * img_with_border[i + 1, j] +
                              g_middle[i, j, 3] * img_with_border[i, j - 1])
            final_image[i - border_width, j - border_width] = outgoing + incoming
    return final_image


def nonlinear_isotropic_diffusion(img_with_border, border_width):
    variance = 50.0
    tau = 0.025
    # --- diffusivity
    gauss = _calculate_diffusivity_matrix(img_with_border, border_width, variance)
    # -- between pixel
    g_middle = _calculate_in_between_diffusivity(gauss, border_width)
    # --- finally ---

    final_image = _diffuse(img_with_border, g_middle, border_width, tau)

    return final_image
