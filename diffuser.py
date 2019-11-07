import numpy as np
from loader import Loader
from numba import njit


class Diffuser:

    @staticmethod
    def nonlinear_isotropic_diffusion(img):
        border_width = 1
        gauss_coeff = 50
        img_with_border = Loader.load_image_with_mirrored_border(img, border_width=border_width)
        gauss = np.zeros(img.shape)
        for i in range(gauss.shape[0]):
            for j in range(gauss.shape[1]):
                pm1_x = img_with_border[i + border_width, j + border_width - 1]
                pp1_x = img_with_border[i + border_width, j + border_width + 1]
                derivative_x = (pp1_x - pm1_x) / float(2)
                pm1_y = img_with_border[i + border_width - 1, j + border_width]
                pp1_y = img_with_border[i + border_width + 1, j + border_width]
                derivative_y = (pp1_y - pm1_y) / float(2)
                grad_square = (derivative_x ** 2) + (derivative_y ** 2)
                gauss_grad = np.exp(-grad_square / gauss_coeff)
                gauss[i, j] = gauss_grad

        # -- between pixel
        g_middle = np.zeros((*gauss.shape, 4))
        for i in range(gauss.shape[0]):
            for j in range(gauss.shape[1]):
                try:
                    # --- top ---
                    g_middle[i, j, 0] = (gauss[i, j] + gauss[i - 1, j]) / 2
                except IndexError:
                    g_middle[i, j, 0] = 0
                try:
                    # --- right ---
                    g_middle[i, j, 1] = (gauss[i, j] + gauss[i, j + 1]) / 2
                except IndexError:
                    g_middle[i, j, 1] = 0
                try:
                    # --- bottom ---
                    g_middle[i, j, 2] = (gauss[i, j] + gauss[i + 1, j]) / 2
                except IndexError:
                    g_middle[i, j, 2] = 0
                try:
                    # --- left ---
                    g_middle[i, j, 3] = (gauss[i, j] + gauss[i, j - 1]) / 2
                except IndexError:
                    g_middle[i, j, 3] = 0

        # --- finally ---
        tau = 0.2
        for i in range(gauss.shape[0]):
            for j in range(gauss.shape[1]):
                outgoing = ((1 - tau) * np.sum(g_middle[i, j])) * img[i, j]
                incoming = 0
                try:
                    # --- top ---
                    incoming += tau * g_middle[i, j, 0] * img[i - 1, j]
                except IndexError:
                    incoming += 0
                try:
                    # --- right ---
                    incoming += tau * g_middle[i, j, 1] * img[i, j + 1]
                except IndexError:
                    incoming += 0
                try:
                    # --- bottom ---
                    incoming += tau * g_middle[i, j, 2] * img[i + 1, j]
                except IndexError:
                    incoming += 0
                try:
                    # --- left ---
                    incoming += tau * g_middle[i,j, 3] * img[i, j-1]
                except IndexError:
                    incoming += 0

                img[i,j] = incoming + outgoing

        return img

    @staticmethod
    def derivative(pm1, pp1, step):
        return (pp1 - pm1) / float(step)
