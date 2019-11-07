from numba import njit
import numpy as np
import cv2

def mirror_image(img_obj, pixels=1):
    img_obj = np.vstack((img_obj, img_obj[-pixels:][:]))
    img_obj = np.vstack((img_obj[:pixels][:], img_obj))
    print(img_obj.shape)
    cv2.imshow("mirror", img_obj)
    img_obj = np.hstack((img_obj[:][:pixels], img_obj))
    img_obj = np.hstack((img_obj, img_obj[:][-pixels:]))
    x, y, _ = img_obj.shape
    print(x, y)
    return img_obj

def nonlinear_isotropic_diffusion(img_obj):
    x, y, _ = img_obj.shape
    # print(x, y)
    img_obj = mirror_image(img_obj, pixels=50)

    # for i in range(x):
    #     for j in range(y):
    #         pass
    # return noisy_lena_img


noisy_img = cv2.imread("data/BoatsNoise10.pgm")
cv2.imshow("test", noisy_img)
nonlinear_isotropic_diffusion(noisy_img)