import cv2
import matplotlib.pyplot as plt
import numpy as np

from ex1_utils import rotate_image, show_flow
from horn_schunck import horn_schunck
from lucas_kanade import lucas_kanade

# ignore if division with zero
np.seterr(divide='ignore', invalid='ignore')


def draw_optical_flow_path(path_1, path_2):
    image_1 = cv2.imread(path_1, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    image_2 = cv2.imread(path_2, cv2.IMREAD_GRAYSCALE).astype(np.float32)

    draw_optical_flow(image_1, image_2)


def draw_optical_flow(image_1, image_2):
    U_lk, V_lk = lucas_kanade(image_1, image_2, 3)
    U_hs, V_hs = horn_schunck(image_1, image_2, 1000, 0.5)

    fig1, ((ax_11, ax_12), (ax_21, ax_22), (ax_31, ax_32)) = plt.subplots(3, 2)

    ax_11.imshow(image_1)
    ax_11.set_title("Image 1")
    ax_12.imshow(image_2)
    ax_12.set_title("Image 2")

    show_flow(U_lk, V_lk, ax_21, type='angle')
    ax_21.set_title("Lucas-Kanade")
    show_flow(U_lk, V_lk, ax_22, type='field', set_aspect=True)
    ax_22.set_title("Lucas-Kanade")

    show_flow(U_hs, V_hs, ax_31, type='angle')
    ax_31.set_title("Horn-Schunk")
    show_flow(U_hs, V_hs, ax_32, type='field', set_aspect=True)
    ax_32.set_title("Horn-Schunk")

    fig1.tight_layout()
    plt.show()


def time_optical_flow(path_1, path_2):
    image_1 = cv2.imread(path_1, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    image_2 = cv2.imread(path_2, cv2.IMREAD_GRAYSCALE).astype(np.float32)

    draw_optical_flow(image_1, image_2)


def draw_optical_flow(image_1, image_2):
    U_lk, V_lk = lucas_kanade(image_1, image_2, 3)
    U_hs, V_hs = horn_schunck(image_1, image_2, 1000, 0.5)


if __name__ == '__main__':
    im1 = np.random.rand(200, 200).astype(np.float32)
    im2 = im1.copy()
    im2 = rotate_image(im2, -1)

    # draw_optical_flow(im1, im2)
    draw_optical_flow_path('disparity/cporta_left.png', 'disparity/cporta_right.png')
    draw_optical_flow_path('disparity/office_left.png', 'disparity/office_right.png')
    draw_optical_flow_path('disparity/office2_left.png', 'disparity/office2_right.png')
