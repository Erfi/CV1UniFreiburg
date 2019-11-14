import numpy as np
import cv2
from typing import Tuple


def minimize_total_variation(img_with_border: np.ndarray, border_width: int) -> Tuple[np.ndarray, list]:
    steps = 10
    sigma = 0.25
    alpha = 0.5
    tau = 0.9
    theta = 1.0
    energy_hist = []
    img = np.zeros(img_with_border.shape)
    rows, cols = img_with_border.shape
    p = np.zeros((rows, cols, 2))
    for i in range(steps):
        if i % 2 == 0:
            print(f'step: {i}')
        grad_u = _calculate_gradient(img_with_border, border_width, vectorize=True)
        _gradient_ascent_p(p, grad_u, border_width, sigma, alpha, vectorize=True)
        _gradient_descent_u(img, img_with_border, p, border_width, tau, theta, vectorize=True)
        energy_hist.append(_calculate_energy(img, img_with_border, grad_u, alpha))
    return img, energy_hist


def _calculate_gradient(img_with_border: np.ndarray, border_width: int, vectorize=True) -> np.ndarray:
    if vectorize:
        sobelx = cv2.Scharr(img_with_border, cv2.CV_64F, 1, 0)
        sobely = cv2.Scharr(img_with_border, cv2.CV_64F, 0, 1)
        grad = np.dstack([sobelx, sobely])
        return grad
    else:
        rows, cols = img_with_border.shape
        grad = np.zeros((rows, cols, 2))
        for i in range(border_width, rows - border_width):
            for j in range(border_width, cols - border_width):
                pm1_x = img_with_border[i, j - 1]
                pp1_x = img_with_border[i, j + 1]
                grad[i, j, 0] = (pp1_x - pm1_x) / 2.0
                pm1_y = img_with_border[i - 1, j]
                pp1_y = img_with_border[i + 1, j]
                grad[i, j, 1] = (pp1_y - pm1_y) / 2.0
        return grad


def _gradient_ascent_p(p: np.ndarray, grad_u: np.ndarray, border_width: int, sigma: float = 0.25,
                       alpha: float = 1.0, vectorize=True) -> None:
    rows, cols = p.shape[0:2]
    if vectorize:
        p += sigma * grad_u
        mag_dual_var = np.linalg.norm(p, axis=2) / alpha
        p[:, :, 0] = p[:, :, 0] / np.maximum(np.ones((rows, cols)), mag_dual_var)
        p[:, :, 1] = p[:, :, 1] / np.maximum(np.ones((rows, cols)), mag_dual_var)
    else:
        for i in range(border_width, rows - border_width):
            for j in range(border_width, cols - border_width):
                p[i, j] = p[i, j] + sigma * grad_u[i, j]
                # --- normalize p to be <= 1 ---
                p_magnitude = np.linalg.norm(p[i, j]) / alpha
                normalizer = np.max(np.array([1.0, p_magnitude]))
                p[i, j] = p[i, j] / normalizer


def _gradient_descent_u(u: np.ndarray, original_img: np.ndarray, p: np.ndarray, border_width: int, tau: float,
                        theta: float, vectorize=True) -> None:
    div_p = _divergence_p(p, border_width, vectorize=vectorize)
    if vectorize:
        new_u = (u + 2 * tau * original_img + tau * div_p) / (1 + 2 * tau)
        # --- optional extra gradient ---
        u[:, :] = u + theta * (new_u - u)

    else:
        rows, cols = u.shape
        for i in range(border_width, rows - border_width):
            for j in range(border_width, cols - border_width):
                old_u = u[i, j]
                u[i, j] = (u[i, j] + 2 * tau * original_img[i, j] + tau * div_p[i, j]) / (1 + 2 * tau)
                # --- optional extra gradient ---
                u[i, j] = u[i, j] + theta * (u[i, j] - old_u)


def _divergence_p(p: np.ndarray, border_width: int, vectorize=True) -> np.ndarray:
    if vectorize:
        sobelx = cv2.Scharr(p[:, :, 0], cv2.CV_64F, 1, 0)
        sobely = cv2.Scharr(p[:, :, 1], cv2.CV_64F, 0, 1)
        div = sobelx + sobely
        return div
    else:
        rows, cols = p.shape[0:2]
        div = np.ones((rows, cols))
        for i in range(border_width, rows - border_width):
            for j in range(border_width, cols - border_width):
                pm1_x = p[i, j - 1, 0]
                pp1_x = p[i, j + 1, 0]
                d_x = (pp1_x - pm1_x) / 2.0
                pm1_y = p[i - 1, j, 1]
                pp1_y = p[i + 1, j, 1]
                d_y = (pp1_y - pm1_y) / 2.0
                div[i, j] = d_x + d_y
        return 25 * div  # it seems like div values must increase to it to work (?)


def _calculate_energy(img, img_with_border, grad_u, alpha):
    energy = (img - img_with_border) ** 2 + (alpha * np.linalg.norm(grad_u, axis=2))
    return np.sum(energy)
