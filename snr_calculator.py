"""
Signal-to-noise ratio calculcator

Author: Mehran Ahkami <ahkami.mehran@gmail.com>
"""
import sys
import math
import time
import numpy as np
import argparse
import image_handler
import add_noise
from numba import njit, cuda


@njit
def calcucalte_psnr(noisy_image: np.ndarray, origin_image: np.ndarray) -> float:
    """Calculcate Peak Signal To Noise of two images
    
    Arguments:
        noisy_image {np.ndarray} -- Image with noise
        origin_image {np.ndarray} -- image without noise 
    
    Returns:
        float -- PSNR
    """
    max_orig = np.amax(origin_image)
    min_orig = np.amin(origin_image)
    row, col = origin_image.shape[:2]
    mean_squere_error = 0
    for i in range(row):
        for j in range(col):
            mean_squere_error += \
                (
                    noisy_image[i][j][0] - origin_image[i][j][0]
                ) ** 2
    psnr = 10 * np.log10(row * ((max_orig - min_orig) ** 2) / mean_squere_error)
    return psnr


if __name__ == "__main__":
    
    if len(sys.argv) < 3:
        print("Two input is required to compute PSNR, noisy and original picture")
        sys.exit()
    parser = argparse.ArgumentParser(description="Function to computer PSNR from two images")
    parser.add_argument("origin", metavar="N", type=str, nargs="+")
    parser.add_argument("noisy", metavar="N", type=str, nargs="+")
    args = parser.parse_args()
    origin_image = image_handler.load_image(args.origin[0])
    noisy_image = image_handler.load_image(args.noisy[0])
    psnr = calcucalte_psnr(noisy_image=noisy_image, origin_image=origin_image)
    print(psnr)
    noisy_images = []
    for i in range(50):
        n_image = add_noise.add_box_muller(origin_image)
        noisy_images.append(n_image)
    t2 = time.time()
    avr_images = np.mean(noisy_images, axis=0)
    image_handler.show_image(noisy_image)
    image_handler.show_image(avr_images)
    psnr = calcucalte_psnr(avr_images, origin_image)
    print(psnr)
