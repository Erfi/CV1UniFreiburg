"""
Add noise to files

Auther: Mehran Ahkami <ahkami.mehran@gmail.com>
"""
import sys
import argparse
import numpy as np
import cv2
import image_handler
from numba import njit, cuda


def add_noise_gaussian(
    image: np.ndarray, variance: float = 1.5, mean: float = 50
) -> np.ndarray:
    """
    Simple method to add gaussian noise to images

    Arguments:
        image {np.ndarray} -- image with a shape of (row, col, ch)

    Keyword Arguments:
        variance {float} -- variance of guassian noise (default: {1.5})
        mean {float} -- mean of guassian noise (default: {50})

    Returns:
        np.ndarray -- noisy image
    """
    row, col = image.shape[:2]
    st_dev = variance ** 2
    noise = np.random.normal(mean, st_dev, (row, col))
    noisy_image = np.zeros(image.shape, np.float32)
    if len(noisy_image.shape) == 2:
        noisy_image = image + noise
    else:
        noisy_image[:, :, 0] = image[:, :, 0] + noise
        noisy_image[:, :, 1] = image[:, :, 1] + noise
        noisy_image[:, :, 2] = image[:, :, 2] + noise
    noisy_image = noisy_image.astype(np.uint8)
    return noisy_image


def add_multiplicative_noise(image: np.ndarray) -> np.ndarray:
    """
    Add multiplicative noise to the image [image = image + n*image]
    and n is a normal distribution of values

    Arguments:
        image {np.ndarray} -- original image

    Returns:
        np.ndarray -- noisy image with multiplicative noise
    """
    row, col = image.shape[:2]
    noise = np.random.randn(row, col)
    noisy_image = np.zeros(image.shape, np.float32)
    if len(noisy_image.shape) == 2:
        noisy_image = image + image * noise
    else:
        noisy_image[:, :, 0] = image[:, :, 0] + image[:, :, 0] * noise
        noisy_image[:, :, 1] = image[:, :, 1] + image[:, :, 0] * noise
        noisy_image[:, :, 2] = image[:, :, 2] + image[:, :, 0] * noise

    noisy_image = noisy_image.astype(np.uint8)
    return noisy_image


def add_impulse_noise(
    image: np.ndarray, amount: float = 0.1, s_vs_p: float = 0.5
) -> np.ndarray:
    """
    Add impulse (salt & pepper) noise to the image
    Arguments:
        image {np.ndarray} -- original image

    Keyword Arguments:
        amount {float} -- amount of pixels to change to 0 or 255
                        (between [0, 1]) (default: {0.1})
        s_vs_p {float} -- Number of white pixels in compare to black pixels
                        (between [0, 1]) (default: {0.5})

    Returns:
        np.ndarray -- noisy image with impluse noise
    """
    row, col = image[:2]
    s_vs_p = s_vs_p
    amount = amount
    noisy_image = np.copy(image)
    # Salt mode
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy_image[tuple(coords)] = 255
    # Pepper mode
    num_pep = np.ceil(amount * image.size * (1.0 - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pep)) for i in image.shape]
    noisy_image[tuple(coords)] = 0
    noisy_image = noisy_image.astype(np.uint8)

    return noisy_image


def add_uniform_noise(image: np.ndarray, percent: float = 0.3) -> np.ndarray:
    """
    Add Uniform noise to the image. Change pixel values
    with random generated number

    Arguments:
        image {np.ndarray} -- original image

    Keyword Arguments:
        percent {float} -- percentage of pixels to be changed (default: {0.3})

    Returns:
        np.ndarray -- noisy image with uniform noise
    """
    row, col = image.shape[:2]
    noise = np.random.randn(row, col)
    noisy_image = np.copy(image)
    num_px = np.ceil(percent * image.size)
    coords = [np.random.randint(0, i - 1, int(num_px)) for i in image.shape]
    noisy_image[tuple(coords)] = noise[tuple(coords[:2])]
    noisy_image = noisy_image.astype(np.uint8)

    return noisy_image


@njit
def add_box_muller(
    image: np.ndarray,
    noise_ratio: float = 25,
    sigma: float = 1.0,
    mean: float = 0.0,  # NOQA
) -> np.ndarray:
    """
    Add bos Muller noise with noise_ratio multiplier

    Arguments:
        image {np.ndarray} -- original image

    Keyword Arguments:
        noise_ratio {float} -- paramater to increase the
        noise in data (default: {25})
        sigma {float} -- standard deviation (default: {1.0})
        mean {float} -- mean (default: {0.0})

    Returns:
        np.ndarray -- noisy image with box muller noise
    """
    row, col = image.shape[:2]
    noisy_image = np.zeros(image.shape, dtype=np.float64)
    for i in range(row):
        for j in range(col):
            u = np.random.rand()
            v = np.random.rand()
            N = np.sqrt(-2 * np.log(u)) * np.cos(2 * np.pi * v)
            M = np.sqrt(-2 * np.log(u)) * np.sin(2 * np.pi * v)  # NOQA
            noisy_image[i][j][0] += noise_ratio * (N * sigma + mean) + image[i][j][0]
            noisy_image[i][j][1] += noise_ratio * (N * sigma + mean) + image[i][j][1]
            noisy_image[i][j][2] += noise_ratio * (N * sigma + mean) + image[i][j][2]
    noisy_image = noisy_image.astype(np.uint8)

    return noisy_image


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("At least one path to image is required")
        sys.exit()
    parser = argparse.ArgumentParser(description="Load image from a path.")
    parser.add_argument("path", metavar="N", type=str, nargs="+")
    parser.add_argument("--window", type=int, nargs="+", default=5000)
    args = parser.parse_args()
    image = image_handler.load_image(file_path=args.path[0])
    noisy_image = add_noise_gaussian(image=image)
    image_handler.show_image(noisy_image, args.window)
    noisy_image = add_multiplicative_noise(noisy_image)
    image_handler.show_image(noisy_image, args.window)
    noisy_image = add_impulse_noise(noisy_image)
    image_handler.show_image(noisy_image, args.window)
    noisy_image = add_uniform_noise(noisy_image)
    image_handler.show_image(noisy_image, args.window)
    noisy_image = add_box_muller(image=image)
    image_handler.show_image(noisy_image, args.window)
    image_handler.save_image(noisy_image)
