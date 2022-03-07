import cv2
import numpy as np
import matplotlib . pyplot as plt

from ex1_utils import rotate_image, show_flow, gaussderiv
from horn_schunck import horn_schunck
from lucas_kanade import lucas_kanade

if __name__ == '__main__':
    
    # im1 = np.random.rand(200, 200).astype(np.float32)
    # im2 = im1.copy()
    # im2 = rotate_image(im2, -1)
    # 
    # (U_lk, V_lk) = lucas_kanade(im1, im2, 3)
    # (U_hs, V_hs) = horn_schunck(im1, im2, 1000, 0.5)
    # 
    # (fig1, ((ax1_11, ax1_12), (ax1_21, ax1_22))) = plt.subplots(2, 2)
    # (fig2, ((ax2_11, ax2_12), (ax2_21, ax2_22))) = plt.subplots(2, 2)
    # 
    # ax1_11.imshow(im1)
    # ax1_12.imshow(im2)
    # 
    # show_flow(U_lk, V_lk, ax1_21, type='angle')
    # show_flow(U_lk, V_lk, ax1_22, type='field', set_aspect=True)
    # 
    # fig1.suptitle('Lucas-Kanade')
    # 
    # ax2_11.imshow(im1)
    # ax2_12.imshow(im2)
    # 
    # show_flow(U_hs, V_hs, ax2_21, type='angle')
    # show_flow(U_hs, V_hs, ax2_22, type='field', set_aspect=True)
    # 
    # fig2.suptitle('Horn-Schunck')

    waffle = cv2.imread('random/test.png')
    
    (fig3, (ax3_11, ax3_12)) = plt.subplots(2, 1)
    Dx, Dy = gaussderiv(waffle, 0.4)
    ax3_11.imshow(Dx)
    ax3_12.imshow(Dy)

    fig3.suptitle('Custom image')
    
    plt.show()
