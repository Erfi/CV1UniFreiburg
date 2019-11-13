import unittest
import numpy as np
import cv2

from loader import Loader
from TVminimizer import minimize_total_variation


class TestMethods(unittest.TestCase):
    def setUp(self) -> None:
        self.filename = "../data/BoatsNoise10.pgm"

    def test_total_variation_minimization(self):
        border_width = 1
        img = cv2.imread(self.filename, cv2.IMREAD_GRAYSCALE)
        img_with_border = Loader.load_image_with_mirrored_border(img, border_width)
        denoised = minimize_total_variation(img_with_border, border_width)
        cv2.imwrite('../results/tv_boat.jpg', denoised)


if __name__ == "__main__":
    unittest.main()
