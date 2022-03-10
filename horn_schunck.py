import cv2
import numpy as np

from ex1_utils import calculate_derivatives


def horn_schunck(img1, img2, n_iters, l):
    """
    img1 - first image matrix (grayscale)
    img2 - second image matrix (grayscale)
    n   - size of the neighbourhood (N x N)
    n_iters − number of iterations (try several hundred)
    lmbd − parameter
    """
    
    i_x, i_y, i_t = calculate_derivatives(img1, img2, 1, 1)
    
    i_x_2 = np.square(i_x)
    i_y_2 = np.square(i_y)
    
    l_d = np.matrix([[0, 0.25, 0], [0.25, 0, 0.25], [0, 0.25, 0]])
    u = np.zeros(img1.shape)
    v = np.zeros(img2.shape)
    
    D = np.add(np.add(l, i_x_2), i_y_2)
    
    for _ in range(n_iters):
        u_a = cv2.filter2D(u, -1, l_d)
        v_a = cv2.filter2D(v, -1, l_d)
        
        P = np.add(
                np.add(
                    np.multiply(i_x, u_a),
                    np.multiply(i_y, v_a)
                ),
                i_t
            )
        
        P_D = np.divide(P, D)
        
        u = np.subtract(
            u_a,
            np.multiply(i_x, P_D)
        )
        
        v = np.subtract(
            v_a,
            np.multiply(i_y, P_D)
        )
    
    return u, v
