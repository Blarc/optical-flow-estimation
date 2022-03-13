import cv2
import numpy as np
from matplotlib import pyplot as plt

from ex1_utils import show_flow
from horn_schunck import horn_schunck
from lucas_kanade import lucas_kanade


def lk_parameter_comparison(img1, img2):
    U1, V1 = lucas_kanade(img1, img2, 10)
    U2, V2 = lucas_kanade(img1, img2, 50)
    U3, V3 = lucas_kanade(img1, img2, 100)

    fig1, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 3))
    ax1.title.set_text('Kernel size: 10x10')
    ax2.title.set_text('Kernel size: 50x50')
    ax3.title.set_text('Kernel size: 100x100')

    show_flow(U1, V1, ax1, type="field", set_aspect=True)
    show_flow(U2, V2, ax2, type="field", set_aspect=True)
    show_flow(U3, V3, ax3, type="field", set_aspect=True)

    plt.tight_layout()
    fig1.savefig(f'./plots/lk_parameter_comparison.png', bbox_inches='tight')
    plt.show()


def hs_parameter_comparison(img1, img2):
    img1 = img1 / 255.0
    img2 = img2 / 255.0

    U1, V1 = horn_schunck(img1, img2, 50, 0.5)
    U2, V2 = horn_schunck(img1, img2, 150, 0.5)
    U3, V3 = horn_schunck(img1, img2, 1000, 0.5)
    U4, V4 = horn_schunck(img1, img2, 300, 0.1)
    U5, V5 = horn_schunck(img1, img2, 300, 1)
    U6, V6 = horn_schunck(img1, img2, 300, 3)

    fig1, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(12, 7))

    ax1.title.set_text('100 iterations, lambda: 0.5')
    ax2.title.set_text('500 iterations, lambda: 0.5')
    ax3.title.set_text('1000 iterations, lambda: 0.5')
    ax4.title.set_text('iterations: 300, lambda: 0.1')
    ax5.title.set_text('iterations: 300, lambda: 1')
    ax6.title.set_text('iterations: 300, lambda: 5')

    show_flow(U1, V1, ax1, type="field", set_aspect=True)
    show_flow(U2, V2, ax2, type="field", set_aspect=True)
    show_flow(U3, V3, ax3, type="field", set_aspect=True)
    show_flow(U4, V4, ax4, type="field", set_aspect=True)
    show_flow(U5, V5, ax5, type="field", set_aspect=True)
    show_flow(U6, V6, ax6, type="field", set_aspect=True)

    plt.tight_layout()
    fig1.savefig(f'./plots/hs_parameter_comparison.png', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    waffle1 = cv2.imread('waffle/waffle1.jpg', cv2.IMREAD_GRAYSCALE).astype(np.float32)
    waffle2 = cv2.imread('waffle/waffle2.jpg', cv2.IMREAD_GRAYSCALE).astype(np.float32)

    # lk_parameter_comparison(waffle1, waffle2)
    hs_parameter_comparison(waffle1, waffle2)