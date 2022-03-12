import cv2
import numpy as np

from ex1_utils import calculate_derivatives
from lucas_kanade import lucas_kanade
from sklearn.metrics.pairwise import cosine_similarity


def horn_schunck(img1, img2, n_iters, lmbd, N=None):
    """
    img1 - first image matrix (grayscale)
    img2 - second image matrix (grayscale)
    n_iters − number of iterations (try several hundred)
    lmbd − parameter
    """

    i_x, i_y, i_t = calculate_derivatives(img1, img2, 1, 1)

    if N:
        u, v = lucas_kanade(img1, img2, N)
    else:
        u, v = np.zeros(img1.shape), np.zeros(img2.shape)

    u_a = np.ones((i_x.shape[0], i_x.shape[1]))
    v_a = np.ones((i_y.shape[0], i_y.shape[1]))
    u_sim, v_sim = 0, 0

    l_d = np.matrix([[0, 0.25, 0], [0.25, 0, 0.25], [0, 0.25, 0]])

    D = np.square(i_x) + np.square(i_y) + lmbd

    matrixElements = i_x.shape[0] * i_x.shape[1]
    
    i = 0
    while i < n_iters and u_sim < 0.6 and v_sim < 0.6:
        u_sim = round(np.sum(cosine_similarity(u, u_a)) / matrixElements, 8)
        v_sim = round(np.sum(cosine_similarity(v, v_a)) / matrixElements, 8)

        if i % 100 == 0:
            print(i, u_sim, v_sim)

        u_a, v_a = cv2.filter2D(u, -1, l_d), cv2.filter2D(v, -1, l_d)

        P = sum([i_t, np.multiply(i_x, u_a), np.multiply(i_y, v_a)])
        P_D = np.divide(P, D)

        u = np.subtract(u_a, np.multiply(i_x, P_D))
        v = np.subtract(v_a, np.multiply(i_y, P_D))

        i = i + 1

    return u, v
