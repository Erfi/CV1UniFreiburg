import numpy as np
from numba import njit


@njit
def minimize_total_variation(img_with_border: np.ndarray, border_width: int) -> np.ndarray:
    steps = 1
    sigma = 0.25
    alpha = 1.0
    tau = 1.0
    img = np.copy(img_with_border)
    p = np.zeros((*img_with_border.shape, 2))
    for i in range(steps):
        grad_u = _calculate_gradient(img_with_border, border_width)
        _gradient_ascent_p(p, grad_u, border_width, sigma, alpha)
        _gradient_descent_u(img, img_with_border, p, border_width, tau)
    return img


@njit
def _calculate_gradient(img_with_border: np.ndarray, border_width: int) -> np.ndarray:
    grad = np.zeros((*img_with_border.shape, 2))
    rows, cols = img_with_border.shape
    for i in range(border_width, rows - border_width):
        for j in range(border_width, cols - border_width):
            pm1_x = img_with_border[i, j + 1]
            pp1_x = img_with_border[i, j - 1]
            grad[i, j, 0] = (pp1_x - pm1_x) / 2.0
            pm1_y = img_with_border[i - 1, j]
            pp1_y = img_with_border[i + 1, j]
            grad[i, j, 1] = (pp1_y - pm1_y) / 2.0
    return grad


@njit
def _gradient_ascent_p(p: np.ndarray, grad_u: np.ndarray, border_width: int, sigma: float = 0.25,
                        alpha: float = 1.0) -> None:
    rows, cols = p.shape[0:2]
    for i in range(border_width, rows - border_width):
        for j in range(border_width, cols - border_width):
            p[i, j] = p[i, j] + sigma * grad_u[i, j]
            # --- normalize p to be <= 1 ---
            p_magnitude = np.linalg.norm(p[i, j]) / alpha
            normalizer = np.max(np.array([1.0, p_magnitude]))
            p[i, j] = p[i, j] / normalizer


@njit
def _gradient_descent_u(u: np.ndarray, original_img: np.ndarray, p: np.ndarray, border_width: int, tau: float) -> None:
    div_p = _divergence_p(p, border_width)
    rows, cols = u.shape
    for i in range(border_width, rows - border_width):
        for j in range(border_width, cols - border_width):
            old_u = u[i, j]
            u[i, j] = (u[i, j] + 2 * tau * original_img[i, j] + tau * div_p[i, j]) / 1.0 + 2 * tau
            # --- optional extra gradient ---
            theta = 1.0
            u[i, j] = u[i, j] + theta * (u[i, j] - old_u)


@njit
def _divergence_p(p: np.ndarray, border_width: int) -> np.ndarray:
    rows, cols = p.shape[0:2]
    div = np.zeros((rows, cols))
    for i in range(border_width, rows - border_width):
        for j in range(border_width, cols - border_width):
            d_x = (p[i, j + 1, 0] - p[i, j - 1, 0]) / 2.0
            d_y = (p[i + 1, j, 1] - p[i - 1, j, 1]) / 2.0
            div[i, j] = d_x + d_y
    return div
