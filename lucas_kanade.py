import numpy as np

from ex1_utils import calculate_derivatives, sum_kernel


def lucas_kanade(img1, img2, N):
    """
    img1 - first image matrix (grayscale)
    img2 - second image matrix (grayscale)
    n   - size of the neighbourhood (N x N)
    """
    i_x, i_y, i_t = calculate_derivatives(img1, img2, 1, 1)

    i_x_t = sum_kernel(np.multiply(i_x, i_t), N)
    i_y_t = sum_kernel(np.multiply(i_y, i_t), N)
    i_x_2 = sum_kernel(np.square(i_x), N)
    i_y_2 = sum_kernel(np.square(i_y), N)
    i_x_y = sum_kernel(np.multiply(i_x, i_y), N)

    D = np.subtract(
        np.multiply(i_x_2, i_y_2),
        np.square(i_x_y)
    )

    D += 1e-15

    u = np.divide(
        np.add(
            np.multiply(-i_y_2, i_x_t),
            np.multiply(i_x_y, i_y_t)
        ),
        D
    )

    v = np.divide(
        np.subtract(
            np.multiply(i_x_y, i_x_t),
            np.multiply(i_x_2, i_y_t)
        ),
        D
    )

    return u, v
