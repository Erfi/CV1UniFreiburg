import unittest
import numpy as np
import cv2

from loader import Loader
from diffuser import nonlinear_isotropic_diffusion


class TestMethods(unittest.TestCase):
    def setUp(self) -> None:
        self.filename = "../data/Boats.pgm"

    def test_load_image_with_border(self):
        img = cv2.imread(self.filename, cv2.IMREAD_GRAYSCALE)
        img_with_border = Loader.load_image_with_mirrored_border(img, 10)
        self.assertIsInstance(img_with_border, np.ndarray)

    def test_nonlinear_isometric_diffusion(self):
        border_width = 1
        img = cv2.imread(self.filename, cv2.IMREAD_GRAYSCALE)
        img_with_border = Loader.load_image_with_mirrored_border(img, border_width)
        for i in range(2):
            diffused = nonlinear_isotropic_diffusion(img_with_border, border_width)
            img_with_border = Loader.load_image_with_mirrored_border(diffused, border_width)
            print(f'ending round {i}')

        cv2.imwrite('../data/diffused.jpeg', diffused)


if __name__ == "__main__":
    unittest.main()
