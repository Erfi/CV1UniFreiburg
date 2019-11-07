"""
class for loading images with different configurations
"""
import cv2
import numpy as np
from numba import njit


class Loader:

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
        return (side_pix + pix) * 0.5

    # @njit
    def nonlinear_isotropic_diffusivity(self, filename, border_width, param, tau, iter):
        it = 0
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        mirrored_img = self.load_image_with_mirrored_border(img, border_width)
        cv2.imwrite('foto.jpg', mirrored_img)

        img_next = np.zeros((img.shape))
        while (it < iter):
            d_x_u = np.zeros((img.shape))
            d_y_u = np.zeros((img.shape))
            # sqr_grad = np.zeros((img.shape))
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    d_x_u[i, j] = self.spatial_derivative(mirrored_img[border_width + i + 1, j + border_width],
                                                          mirrored_img[border_width + i - 1, border_width + j])
                    d_y_u[i, j] = self.spatial_derivative(mirrored_img[border_width + i, border_width + j + 1],
                                                          mirrored_img[border_width + i, border_width + j - 1])

            abs_sqr_grad = d_x_u * d_x_u + d_y_u * d_y_u
            discret_grad = self.exponential_diffusivity(abs_sqr_grad, param)
            diff_left = np.zeros((img.shape))
            diff_right = np.zeros((img.shape))
            diff_buttom = np.zeros((img.shape))
            diff_top = np.zeros((img.shape))

            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    diff_buttom[i, j] = self.diff_between_edges(mirrored_img[border_width + i, border_width + j + 1],
                                                                mirrored_img[border_width + i, border_width + j])
                    diff_top[i, j] = self.diff_between_edges(mirrored_img[border_width + i, border_width + j - 1],
                                                             mirrored_img[border_width + i, border_width + j])
                    diff_right[i, j] = self.diff_between_edges(mirrored_img[border_width + i + 1, border_width + j],
                                                               mirrored_img[border_width + i, border_width + j])
                    diff_left[i, j] = self.diff_between_edges(mirrored_img[border_width + i - 1, border_width + j],
                                                              mirrored_img[border_width + i, border_width + j])

            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    img_next[i, j] = (
                                (1 - tau * (diff_buttom[i, j] + diff_top[i, j] + diff_right[i, j] + diff_left[i, j])) *
                                img[i, j] +
                                tau * (diff_buttom[i, j] * mirrored_img[i, j + 1] + diff_top[i, j] * mirrored_img[
                            i, j - 1] +
                                       diff_right[i, j] * mirrored_img[i + 1, j] + diff_left[i, j] * mirrored_img[
                                           i - 1, j]))
            img = img_next
            it = it + 1
        return img


filename = "../data/Boats.pgm"
loader_obj = Loader()
result = loader_obj.nonlinear_isotropic_diffusivity(filename, 10, 50, 0.1, 2)
print(result)
cv2.imshow('image', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
