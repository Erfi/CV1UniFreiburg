import unittest
from loader import Loader
import cv2


class TestMethods(unittest.TestCase):

    def test_load_image_with_border(self):
        filename = "../data/head.pgm"
        img = Loader.load_image_with_mirrored_border(filename, 0, cv2.IMREAD_GRAYSCALE)
        #
        # cv2.imshow('image', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


if __name__ == "__main__":
    unittest.main()

