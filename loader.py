"""
class for loading images with different configurations
"""
import cv2
import numpy as np


class Loader:

    @staticmethod
    def load_image_with_mirrored_border(filename, *args):
        img = cv2.imread(filename, *args)
        return img

    def mirror_image(img_obj, pixels=1):
        img_obj = np.vstack([img_obj, img_obj[-pixels:][:]])
        img_obj = np.vstack([img_obj[:pixels][:], img_obj])
        img_obj = np.hstack([img_obj[:][:pixels], img_obj])
        img_obj = np.hstack([img_obj, img_obj[:][-pixels:]])
        return img_obj
