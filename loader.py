"""
class for loading images with different configurations
"""
import cv2
import numpy as np
from numba import njit


class Diffuser:

    def load_image_with_mirrored_border(self, img, border_width):

        if border_width == 0:
            return img

        rows, cols = img.shape
        left_border = np.flip(img[:, 0:border_width], axis=1)
        right_border = np.flip(img[:, cols - border_width: cols], axis=1)
        top_border = np.flip(img[0:border_width, :], axis=0)
        bottom_border = np.flip(img[rows - border_width: rows, :], axis=0)

        result = np.zeros((rows + 2 * border_width, cols + 2 * border_width), np.int)
        result[border_width: rows + border_width, border_width:cols + border_width] = img
        result[border_width: rows + border_width, 0: border_width] = left_border
        result[border_width: rows + border_width, cols + border_width: cols + 2 * border_width] = right_border
        result[0: border_width, border_width: cols + border_width] = top_border
        result[rows + border_width: rows + 2 * border_width, border_width:cols + border_width] = bottom_border

        return result

    def exponential_diffusivity(self, x, param):
        x_scaled = -1. / (param ** 2) * x
        return np.exp(x_scaled)

    def spatial_derivative(self, up, un):
        return (up - un) * 0.5

    def diff_between_edges(self, side_pix, pix):
        return (side_pix + pix)*0.5

    #@njit
    def _grad_diffusivity_matrix(self, mirrored_img, param):
        d_x_u = np.zeros((mirrored_img.shape))
        d_y_u = np.zeros((mirrored_img.shape))
        for i in range(1, (mirrored_img.shape[0] - 1), 1):
            for j in range(1, (mirrored_img.shape[1] - 1), 1):
                d_x_u[i, j] = self.spatial_derivative(mirrored_img[i + 1, j],
                                                      mirrored_img[i - 1, j])
                d_y_u[i, j] = self.spatial_derivative(mirrored_img[i, j + 1],
                                                      mirrored_img[i, j - 1])
        abs_sqr_grad = np.multiply(d_x_u, d_x_u) + np.multiply(d_y_u, d_y_u)
        discret_grad = self.exponential_diffusivity(abs_sqr_grad, param)
        return discret_grad

    #@njit
    def _diffuse(self, discret_grad):
        diffus_edges = np.zeros((*discret_grad.shape, 4))
        for i in range(1, (diffus_edges.shape[0] - 1), 1):
            for j in range(1, (diffus_edges.shape[1] - 1), 1):
                # --Bottom--
                diffus_edges[i, j, 0] = self.diff_between_edges(discret_grad[i, j + 1],
                                                                discret_grad[i, j])
                # --Top--
                diffus_edges[i, j, 1] = self.diff_between_edges(discret_grad[i, j],
                                                                discret_grad[i, j - 1])
                # --Right--
                diffus_edges[i, j, 2] = self.diff_between_edges(discret_grad[i + 1, j],
                                                                discret_grad[i, j])
                # --Left--
                diffus_edges[i, j, 3] = self.diff_between_edges(discret_grad[i, j],
                                                                discret_grad[i - 1, j])
        return diffus_edges

    #@njit
    def nonlinear_isotropic_diffusivity(self, filename, border_width, param, tau, iter):
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        mirrored_img = self.load_image_with_mirrored_border(img, border_width)
        img_next = np.zeros((mirrored_img.shape))
        for it in range(iter):
            discret_grad = self._grad_diffusivity_matrix(mirrored_img, param)
            diffus_edges = self._diffuse(discret_grad)
            for i in range(1, (mirrored_img.shape[0] - 1), 1):
                for j in range(1, (mirrored_img.shape[1] - 1), 1):
                    img_next[i, j] = (
                                (1 - tau * (diffus_edges[i, j, 0] + diffus_edges[i, j, 1] + diffus_edges[i, j, 2] + diffus_edges[i, j, 3])) *
                                mirrored_img[i, j] +
                                tau * (diffus_edges[i, j, 0] * mirrored_img[i, j + 1] + diffus_edges[i, j, 1] * mirrored_img[
                            i, j - 1] +
                                       diffus_edges[i, j, 2] * mirrored_img[i + 1, j] + diffus_edges[i, j, 3] * mirrored_img[
                                           i - 1, j]))
            mirrored_img = img_next
        return mirrored_img


filename = "../data/BoatsNoise10.pgm"
diffused_obj = Diffuser()
result = diffused_obj.nonlinear_isotropic_diffusivity(filename, 5, 30, 0.001, 5)
cv2.imwrite('foto_denoised_lambda=30_t=0.1.jpg', result)
