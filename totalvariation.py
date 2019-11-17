import cv2
import numpy as np
import matplotlib.pyplot as plt


class TVminimizer:

    def __init__(self, alpha: float, sigma: float, theta: float,
                                steps: int, tau: float,
                                filename: str, border_width: int):
        self.alpha = alpha
        self.sigma = sigma
        self.tau = tau
        self.theta = theta
        self.steps = steps
        self.border_width = border_width
        self.img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

    def load_image_with_mirrored_border(self, img: np.ndarray) -> np.ndarray:

        if self.border_width == 0:
            return img

        rows, cols = img.shape
        left_border = np.flip(img[:, 0:self.border_width], axis=1)
        right_border = np.flip(img[:, cols - self.border_width: cols], axis=1)
        top_border = np.flip(img[0:self.border_width, :], axis=0)
        bottom_border = np.flip(img[rows - self.border_width: rows, :], axis=0)

        result = np.zeros((rows + 2 * self.border_width, cols + 2 * self.border_width), np.int)
        result[self.border_width: rows + self.border_width, self.border_width:cols + self.border_width] = img
        result[self.border_width: rows + self.border_width, 0: self.border_width] = left_border
        result[self.border_width: rows + self.border_width, cols + self.border_width: cols + 2 * self.border_width] = right_border
        result[0: self.border_width, self.border_width: cols + self.border_width] = top_border
        result[rows + self.border_width: rows + 2 * self.border_width, self.border_width:cols + self.border_width] = bottom_border

        return result

    def projected_gradient_ascent_dual(self, p: np.ndarray, grad_u: np.ndarray) -> np.ndarray:
        #print("Projecting the Gradient ascent over dual variable...")
        num = p + self.sigma*grad_u
        den = num
        den = np.where((abs(num)/self.alpha) < 1, num, 1)
        return num/den

    def gradient_descent_primal(self, p: np.ndarray, u: np.ndarray) -> np.ndarray:
        #print("Computing Gradient descent primal variable...")
        num = u + 2*self.tau*self.img_orig + self.tau*self.divergency_dual(p)
        den = 1 + 2*self.tau
        u_k = num/den
        u_k_hat = u_k + self.theta*(u_k - u)
        return u_k_hat

    def divergency_dual(self, p : np.ndarray) -> np.ndarray:
        #print("Calculating divergency over dual variable...")
        rows, cols = p.shape[0], p.shape[1]
        div = np.zeros((rows, cols, 2))
        for i in range(self.border_width, rows - self.border_width):
            for j in range(self.border_width, cols - self.border_width):
                pm1_x = p[i, j - 1, 0]
                pp1_x = p[i, j + 1, 0]
                div[i, j, 0] = (pp1_x - pm1_x) / 2.0
                pm1_y = p[i - 1, j, 1]
                pp1_y = p[i + 1, j, 1]
                div[i, j, 1] = (pp1_y - pm1_y) / 2.0
        div = div[:, :, 0] + div[:, :, 1]
        div.reshape(rows, cols)
        return div

    def calculate_spatial_gradient(self, u: np.ndarray) -> np.ndarray:
        #print("computing spatial gradient...")
        rows, cols = u.shape
        grad = np.zeros((rows, cols, 2))
        for i in range(self.border_width, rows - self.border_width):
            for j in range(self.border_width, cols - self.border_width):
                pm1_x = u[i, j - 1]
                pp1_x = u[i, j + 1]
                grad[i, j, 0] = (pp1_x - pm1_x) / 2.0
                pm1_y = u[i - 1, j]
                pp1_y = u[i + 1, j]
                grad[i, j, 1] = (pp1_y - pm1_y) / 2.0

        return grad

    def compute_energy(self, u) -> float:
        grad_u = self.calculate_spatial_gradient(u)
        energy = np.sum((u - self.img_orig) ** 2 + self.alpha * np.linalg.norm(grad_u, axis=2))
        print(energy)
        return energy

    def minimize_total_variation(self):

        self.img_orig = self.load_image_with_mirrored_border(self.img)
        print(self.img_orig.shape)
        u_hat = np.zeros(self.img_orig.shape)
        rows, cols = self.img_orig.shape
        p = np.ones((rows, cols, 2))
        energy_history = []
        for i in range(self.steps):
            grad_u = self.calculate_spatial_gradient(u_hat)
            #print("Shape of gradient of primal variable u: "+str(grad_u.shape))
            p = self.projected_gradient_ascent_dual(p, grad_u)
            #print("Shape of dual variable p after projection: "+str(p.shape))
            u_hat = self.gradient_descent_primal(p, u_hat)
            energy_history.append(self.compute_energy(u_hat))
            if i % 10 == 0:
                print(f'step: {i}')
        print("End of process...")
        return u_hat, energy_history

filename = "/home/mikel/Desktop/Computer Science Master/CV/ex_1/data/BoatsNoise10.pgm"
steps = 100
diffused_obj = TVminimizer(alpha=100 , sigma=0.025, theta=1.5,
                           steps=steps, tau=0.01, filename=filename,
                           border_width=10)
result, history_list = diffused_obj.minimize_total_variation()
x = np.arange(0, steps, 1)
plt.plot(x,history_list)
plt.show()
cv2.imwrite('foto_after_TMmin.jpg', result)





