from numba import jit, njit, cuda
import cv2
import numpy as np
import time

# @njit(parallel=True, nogil=True)
def primal_dual(img_obj, original_img, original_shape):
    x, y = original_shape
    dual_var = np.ones((x, y, 2))
    # gradient = np.zeros(dual_var.shape)
    sigma = 0.25
    theta = 1
    tau = 0.9
    alpha = 10
    iterations = 200
    previous_energy = 10000000000
    for k in range(iterations):
        sobelx = np.hstack((img_obj[:, 1:], np.zeros((x, 1)))) - np.hstack((np.zeros((x, 1)), img_obj[:, :-1]))
        sobely = np.vstack((img_obj[1:, :], np.zeros((1, y)))) - np.vstack((np.zeros((1, y)), img_obj[:-1, :]))
        gradient = np.dstack((sobelx, sobely))
        if k == 0:
            dual_var = gradient
        dual_var += sigma * gradient
        dual_var[:, :, 0] = dual_var[:, :, 0] / np.maximum(np.ones(original_shape), np.sqrt(
            np.square(dual_var[:, :, 0]) + np.square(dual_var[:, :, 1])) / alpha)
        dual_var[:, :, 1] = dual_var[:, :, 1] / np.maximum(np.ones(original_shape), np.sqrt(
            np.square(dual_var[:, :, 0]) + np.square(dual_var[:, :, 1])) / alpha)

        sobelx_dual = np.hstack((dual_var[:, 1:, 0], np.zeros((x, 1)))) - np.hstack(
            (np.zeros((x, 1)), dual_var[:, :-1, 0]))
        sobely_dual = np.vstack((dual_var[1:, :, 1], np.zeros((1, y)))) - np.vstack(
            (np.zeros((1, y)), dual_var[:-1, :, 1]))
        divergence_dual_var = sobelx_dual + sobely_dual
        new_img_obj = (img_obj + 2 * tau * original_img + tau * divergence_dual_var) / (1 + 2 * tau)
        new_img_obj = img_obj + theta * (new_img_obj - img_obj)
        energy = (img_obj - original_img) ** 2 + (alpha * (np.square(gradient[:, :, 0]) + np.square(gradient[:, :, 1])))
        # if previous_energy > energy.sum():
        #     previous_energy = energy.sum()
        img_obj = new_img_obj
        # print("Energy is %f", energy.sum())
    return img_obj


tick = time.time()
img = cv2.imread("data/BoatsNoise10.pgm", cv2.IMREAD_GRAYSCALE)
img = img/1.0
shape = img.shape
img = primal_dual(img, img.copy(), shape)
cv2.imwrite("data/BoatsProcessedPrimalDual.jpg", img)
tock = time.time()
print("Total time elapsed %f", tock - tick)
