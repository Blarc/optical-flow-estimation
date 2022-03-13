import cv2
import matplotlib.pyplot as plt
import numpy as np

from ex1_utils import rotate_image, show_flow
from horn_schunck import horn_schunck
from lucas_kanade import lucas_kanade

# ignore if division with zero
np.seterr(divide='ignore', invalid='ignore')


def draw_optical_flow_path(path_1, path_2, filename='test'):
    image_1 = cv2.imread(path_1, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    image_2 = cv2.imread(path_2, cv2.IMREAD_GRAYSCALE).astype(np.float32)

    draw_optical_flow(image_1, image_2, filename=filename)


def draw_optical_flow(image_1, image_2, type_='field', normalizeImages=True, filename='test'):

    if normalizeImages:
        image_1 = image_1 / 255.0
        image_2 = image_2 / 255.0
        
    U_lk, V_lk = lucas_kanade(image_1, image_2, 10)
    U_hs, V_hs = horn_schunck(image_1, image_2, 10000, 0.5, 10)

    fig1, ((ax_11, ax_12), (ax_21, ax_22)) = plt.subplots(2, 2)

    ax_11.imshow(image_1)
    ax_11.set_title("Frame t")
    ax_12.imshow(image_2)
    ax_12.set_title("Frame t + 1")

    show_flow(U_lk, V_lk, ax_21, type=type_, set_aspect=True)
    ax_21.set_title("Lucas-Kanade")

    show_flow(U_hs, V_hs, ax_22, type=type_, set_aspect=True)
    ax_22.set_title("Horn-Schunck")

    fig1.tight_layout()
    fig1.savefig(f'./plots/{filename}.png', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    im1 = np.random.rand(200, 200).astype(np.float32)
    im2 = im1.copy()
    im2 = rotate_image(im2, -1)

    # draw_optical_flow(im1, im2, filename='random_noise')
    # draw_optical_flow_path('disparity/office2_left.png', 'disparity/office2_right.png', filename='office_2')
    # draw_optical_flow_path('lab2/001.jpg', 'lab2/002.jpg', filename='lab_2')
    # draw_optical_flow_path('collision/00000060.jpg', 'collision/00000061.jpg', filename='collision')
    draw_optical_flow_path('waffle/waffle1.jpg', 'waffle/waffle2.jpg', filename='waffle')
    # draw_optical_flow_path('waffle/waffle1fast.jpg', 'waffle/waffle2fast.jpg', filename='waffle_fast')
