import cv2
import math
import time
from numba import cuda, njit

import numpy as np

start = time.time()


# def box_mueller():
#     u = np.random.rand()
#     v = np.random.rand()
#     n = math.sqrt(-3 * np.log(u)) * np.cos(2 * np.pi * v)
#     m = math.sqrt(-3 * np.log(v)) * np.sin(2 * np.pi * u)
#     return n, m


# @njit
def calculate_psnr(grnd_truth, noisy):
    max_grnd = np.amax(grnd_truth)
    min_grnd = np.amin(grnd_truth)
    x, y, _ = grnd_truth.shape
    mean_square_error = 0
    for i in range(x):
        for j in range(y):
            mean_square_error += \
                (noisy[i][j][0] - grnd_truth[i][j][0]) ** 2
    if mean_square_error == 0:
        return 0
    psnr = 10 * np.log10((x * ((max_grnd - min_grnd) ** 2)) / mean_square_error)
    return psnr


@njit
def add_noise(lena_img, noise_ratio):
    x, y, _ = lena_img.shape
    noisy_lena_img = np.empty(lena_img.shape)
    for i in range(x):
        for j in range(y):
            u = np.random.rand()
            v = np.random.rand()
            n = math.sqrt(-3 * np.log(u)) * np.cos(2 * np.pi * v)
            m = math.sqrt(-3 * np.log(v)) * np.sin(2 * np.pi * u)
            m = m * noise_ratio
            noisy_lena_img[i][j] = [max(0, min(lena_img[i][j][0] + m, 255))] * 3
    return noisy_lena_img

def add_noise_without_GPU(lena_img, noise_ratio):
    x, y, _ = lena_img.shape
    noisy_lena_img = np.empty(lena_img.shape)
    for i in range(x):
        for j in range(y):
            u = np.random.rand()
            v = np.random.rand()
            n = math.sqrt(-3 * np.log(u)) * np.cos(2 * np.pi * v)
            m = math.sqrt(-3 * np.log(v)) * np.sin(2 * np.pi * u)
            m = m * noise_ratio
            noisy_lena_img[i][j] = [max(0, min(lena_img[i][j][0] + m, 255))] * 3
    return noisy_lena_img

lena_img = cv2.imread("lena.pgm")
noisy_lena_img = add_noise(lena_img, 50)
psnr = calculate_psnr(lena_img, noisy_lena_img)
# print(psnr)
cv2.imwrite("lena_noisy.jpg", noisy_lena_img)
averaged_noisy_img = np.zeros(lena_img.shape)
for i in range(50):
    averaged_noisy_img += add_noise_without_GPU(lena_img, 10)
psnr = calculate_psnr(lena_img, averaged_noisy_img/50)
# print(psnr)
time_consumed = time.time() - start
print("Total time consumed without GPU %s", time_consumed)

start = time.time()
averaged_noisy_img = np.zeros(lena_img.shape)
for i in range(50):
    averaged_noisy_img += add_noise(lena_img, 10)
psnr = calculate_psnr(lena_img, averaged_noisy_img/50)
# print(psnr)
time_consumed = time.time() - start
print("Total time consumed with GPU %s", time_consumed)