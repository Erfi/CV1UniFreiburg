import unittest
from matplotlib import pyplot as plt
import cv2

from loader import Loader
from TVminimizer import minimize_total_variation


class TestMethods(unittest.TestCase):
    def setUp(self) -> None:
        self.filename = "../data/BoatsNoise10.pgm"
        self.image_name = self.filename.split('/')[-1].split('.')[0].strip()

    def test_total_variation_minimization(self):
        border_width = 1
        img = cv2.imread(self.filename, cv2.IMREAD_GRAYSCALE)
        img_with_border = Loader.load_image_with_mirrored_border(img, border_width)
        denoised, hist = minimize_total_variation(img_with_border, border_width)
        # --- save denoised image ---
        cv2.imwrite(f'../results/tv_{self.image_name}.jpg', denoised)
        # --- save energy plot ---
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.plot(hist)
        ax.set_title('Total Variation Energy After Each Iteration')
        ax.set_xlabel('iteration number')
        ax.set_ylabel('energy')
        fig.savefig(f'../results/tv_{self.image_name}_energy.png')


if __name__ == "__main__":
    unittest.main()
