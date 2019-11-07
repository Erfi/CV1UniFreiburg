"""
class for loading images with different configurations
"""
import cv2
import numpy as np


class Loader:

    @staticmethod
    def load_image_with_mirrored_border(filename, border_width, *args):
        img = cv2.imread(filename, *args)
        if border_width == 0:
            return img

        rows, cols = img.shape
        left_border = np.flip(img[:, 0:border_width], axis=1)
        right_border = np.flip(img[:, cols - border_width: cols], axis=1)
        top_border = np.flip(img[0:border_width, :], axis=0)
        bottom_border = np.flip(img[rows - border_width: rows, :], axis=0)

        result = np.zeros((rows + 2 * border_width, cols + 2 * border_width), np.uint8)
        result[border_width: rows + border_width, border_width:cols + border_width] = img
        result[border_width: rows + border_width, 0: border_width] = left_border
        result[border_width: rows + border_width, cols + border_width: cols + 2 * border_width] = right_border
        result[0: border_width, border_width: cols + border_width] = top_border
        result[rows + border_width: rows + 2 * border_width, border_width:cols + border_width] = bottom_border

        return result

    def mirror_image(img_obj, pixels=1):
        img_obj = np.vstack([img_obj, img_obj[-pixels:][:]])
        img_obj = np.vstack([img_obj[:pixels][:], img_obj])
        img_obj = np.hstack([img_obj[:][:pixels], img_obj])
        img_obj = np.hstack([img_obj, img_obj[:][-pixels:]])
        return img_obj
