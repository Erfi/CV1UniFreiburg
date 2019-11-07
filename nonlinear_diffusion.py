from numba import njit
import numpy as np


def nonlinear_isotropic_diffusion(img_obj):
    x, y, _ = img_obj.shape
    print(x, y)
    img_obj = mirror_image(img_obj, pixels=5)
    # for i in range(x):
    #     for j in range(y):
    #         pass
    # return noisy_lena_img

#
# nonlinear_isotropic_diffusion()
