import cv2
import matplotlib.pyplot as plot
import numpy as np
import os
import time


def primal_dual(original_img):
    original_shape = original_img.shape
    x, y = original_shape
    dual_var = np.ones((x, y, 2))
    x_half = int(x / 2)
    segmented_img = np.vstack((np.ones((x_half, y)), np.zeros((x - x_half, y))))
    weight = 1
    sigma = 0.25
    theta = 1
    tau = 0.1
    nu = 0.09
    iterations = 1000
    energy_map = {}
    for k in range(iterations):
        grad_x = np.hstack((segmented_img[:, 1:], segmented_img[:, -2:-1])) - segmented_img
        grad_y = np.vstack((segmented_img[1:, :], segmented_img[-2:-1, :])) - segmented_img
        gradient = np.dstack((grad_x, grad_y))
        if k == 0:
            dual_var = gradient
        dual_var += sigma * weight * gradient
        dual_var[:, :, 0] = dual_var[:, :, 0] / np.maximum(np.ones(original_shape), np.sqrt(
            np.square(dual_var[:, :, 0]) + np.square(dual_var[:, :, 1])) / nu)
        dual_var[:, :, 1] = dual_var[:, :, 1] / np.maximum(np.ones(original_shape), np.sqrt(
            np.square(dual_var[:, :, 0]) + np.square(dual_var[:, :, 1])) / nu)
        grad_x_dual = weight * (np.hstack((dual_var[:, 1:, 0], np.zeros((x, 1)))) - np.hstack(
            (np.zeros((x, 1)), dual_var[:, :-1, 0])))
        grad_y_dual = weight * (np.vstack((dual_var[1:, :, 1], np.zeros((1, y)))) - np.vstack(
            (np.zeros((1, y)), dual_var[:-1, :, 1])))
        divergence_dual_var = grad_x_dual + grad_y_dual
        mean_segmented = np.mean(segmented_img)
        mu_1 = np.argwhere(segmented_img > mean_segmented).mean(0).astype(int)
        mu_2 = np.argwhere(segmented_img < mean_segmented).mean(0).astype(int)
        psi = (original_img - original_img[mu_1[0], mu_1[1]]) ** 2 - (
                    original_img - original_img[mu_2[0], mu_2[1]]) ** 2
        new_segmented_img = segmented_img - (tau * (psi/nu - divergence_dual_var))
        new_segmented_img = segmented_img + (theta * (new_segmented_img - segmented_img))
        energy = np.multiply(psi, new_segmented_img) + nu * weight * np.sqrt(
            np.square(gradient[:, :, 0]) + np.square(gradient[:, :, 1]))
        energy_map[energy.sum()] = new_segmented_img
        segmented_img = new_segmented_img
    plot.plot(list(range(len(energy_map))), list(energy_map.keys()))
    plot.show()
    return energy_map[min(energy_map)]


images = ["withHolesHarder.pgm"]
root_directory = "data/segmentation"
for image in images:
    tick = time.time()
    img_path = os.path.join(root_directory, image)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    original_img = img.copy()
    img = img/255.0
    shape = img.shape
    processed_img = primal_dual(img)
    mean = np.mean(processed_img)
    processed_img[processed_img > mean] = 255
    processed_img[processed_img != 255] = 0
    target_img_path = os.path.join(root_directory, "tv_primal_dual_" + os.path.splitext(image)[0] + ".jpg")
    juxtaposed_img = np.hstack((original_img, processed_img))
    cv2.imwrite(target_img_path, juxtaposed_img)
    tock = time.time()
    print("Total time elapsed %f for %s" % (tock - tick, image))
#