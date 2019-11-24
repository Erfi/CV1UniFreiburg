import unittest
from matplotlib import pyplot as plt
import cv2

from loader import Loader
from tvsegmenter import minimize_total_variation


class TestMethods(unittest.TestCase):
    def setUp(self) -> None:
        self.filename = "../data/coins.pgm"
        self.image_name = self.filename.split('/')[-1].split('.')[0].strip()

    def test_total_variation_minimization(self):
        border_width = 1
        img = cv2.imread(self.filename, cv2.IMREAD_GRAYSCALE)
        img_with_border = Loader.load_image_with_mirrored_border(img, border_width)
        segmented, hist = minimize_total_variation(img_with_border, border_width)

        # --- save segmented image ---
        cv2.imwrite(f'../results/tv_segmented_{self.image_name}.jpg', segmented)
        # --- save energy plot ---
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.plot(hist)
        ax.set_title('Total Variation Energy After Each Iteration')
        ax.set_xlabel('iteration number')
        ax.set_ylabel('energy')
        fig.savefig(f'../results/tv_segmented_{self.image_name}_energy.png')


if __name__ == "__main__":
    unittest.main()
