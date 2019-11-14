from numba import jit, njit, cuda
import cv2
import numpy as np

from loader import Loader



def primal_dual(img_obj, original_img, original_shape):
    x, y = original_shape
    dual_var = np.ones((x, y, 2))
    gradient = np.zeros(dual_var.shape)
    sigma = 0.25
    theta = 1
    tau = 0.9
    alpha = 1
    lt = 1
    iterations = 50
    previous_energy = 10000000000
    for k in range(iterations):
        sobelx = cv2.Scharr(original_img, cv2.CV_64F, 1, 0)
        sobely = cv2.Scharr(original_img, cv2.CV_64F, 0, 1)
        gradient = np.dstack([sobelx, sobely])
        if k == 0:
            dual_var = gradient
        dual_var += sigma * gradient
        dual_var[:, :, 0] = dual_var[:, :, 0]/np.maximum(np.ones(original_shape), np.sqrt(np.square(dual_var[:, :, 0]) + np.square(dual_var[:, :, 1]))/alpha)
        dual_var[:, :, 1] = dual_var[:, :, 1]/np.maximum(np.ones(original_shape), np.sqrt(np.square(dual_var[:, :, 0]) + np.square(dual_var[:, :, 1]))/alpha)

        sobelx_dual = cv2.Scharr(dual_var[:, :, 0], cv2.CV_64F, 1, 0)
        sobely_dual = cv2.Scharr(dual_var[:, :, 1], cv2.CV_64F, 0, 1)
        divergence_dual_var = sobelx_dual + sobely_dual
        new_img_obj = (img_obj + 2 * tau * original_img + tau * divergence_dual_var)/(1 + 2 * tau)
        new_img_obj = img_obj + theta * (new_img_obj - img_obj)
        energy = (img_obj - original_img) ** 2 + (alpha * (np.square(gradient[:, :, 0]) + np.square(gradient[:, :, 1])))
        if previous_energy > energy.sum():
            previous_energy = energy.sum()
            img_obj = new_img_obj
    return img_obj


img_obj = cv2.imread("data/BoatsNoise10.pgm", cv2.IMREAD_GRAYSCALE)
shape = img_obj.shape
original_img = img_obj.copy()
img_obj = primal_dual(img_obj, original_img, shape)
cv2.imwrite("data/BoatsProcessedPrimalDual.jpg", img_obj)
