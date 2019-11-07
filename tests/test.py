import unittest
import numpy as np
import cv2

from loader import Loader
from diffuser import Diffuser


class TestMethods(unittest.TestCase):
    def setUp(self) -> None:
        self.filename = "../data/Boats.pgm"

    def test_load_image_with_border(self):
        img = cv2.imread(self.filename, cv2.IMREAD_GRAYSCALE)
        img_with_border = Loader.load_image_with_mirrored_border(img, 10)
        self.assertIsInstance(img_with_border, np.ndarray)

    def test_nonlinear_isometric_diffusion(self):
        img = cv2.imread(self.filename, cv2.IMREAD_GRAYSCALE)
        diffuser = Diffuser()
        for i in range(1):
            img = diffuser.nonlinear_isotropic_diffusion(img)
            print(f'ending round {i}')

        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    unittest.main()

