"""
class for loading images with different configurations
"""
import os
import sys
import cv2
import numpy as np


class ImageHandler:
    def __init__(self, border=(1, 1)):
        self.border = border
        self.orig_shape = []

    def load_image(self, image_path, load_type):
        img = cv2.imread(image_path, load_type)
        self.orig_shape = img.shape
        if self.border == (0, 0):
            return img.astype("float")
        if img.ndim == 3:
            img_pad = np.pad(img, (self.border, self.border, (0, 0)), mode="edge")
            return img_pad.astype("float")
        if img.ndim == 2:
            img_pad = np.pad(img, (self.border, self.border), "edge")
            return img_pad.astype("float")
        return 0

    def get_originalShape(self):
        return self.orig_shape

    def save_image(self, image: np.ndarray, name: str = "data/noisy_image.jpg") -> None:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        print(os.getcwd())
        path = os.path.join(dir_path, name)
        if os.path.exists(os.path.join(dir_path, "data")):
            try:
                cv2.imwrite(path, image)
            except Exception:
                print("There is an error in saving image")
                sys.exit()
        else:
            os.mkdir(os.path.join(dir_path, "data"))
            try:
                cv2.imwrite(path, image)
            except Exception:
                print("There is an error in saving image")
                sys.exit()

    def show_image(self, image: np.ndarray, window_time: int = 2500) -> None:
        cv2.imshow("image", image)
        cv2.waitKey(window_time)  # in ms
        cv2.destroyAllWindows()
