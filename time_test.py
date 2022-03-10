import time

import cv2
import numpy as np

from horn_schunck import horn_schunck
from lucas_kanade import lucas_kanade

# ignore if division with zero
np.seterr(divide='ignore', invalid='ignore')


def time_optical_flow_path(path_1, path_2):
    image_1 = cv2.imread(path_1, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    image_2 = cv2.imread(path_2, cv2.IMREAD_GRAYSCALE).astype(np.float32)

    time_optical_flow(image_1, image_2)


def time_optical_flow(image_1, image_2):
    lk_start_time = time.time()
    lucas_kanade(image_1, image_2, 3)
    lk_time = time.time() - lk_start_time

    hs_start_time = time.time()
    horn_schunck(image_1, image_2, 1000, 0.5)
    hs_time = time.time() - hs_start_time

    lk_hs_start_time = time.time()
    horn_schunck(image_1, image_2, 1000, 0.5, 3)
    lk_hs_time = time.time() - lk_hs_start_time

    print('-' * 75)
    print(f'Lucas-Kanade:                      {lk_time}')
    print(f'Horn-Shunck:                       {hs_time}')
    print(f'Horn-Shunck (initalized with l-k): {lk_hs_time}')


if __name__ == '__main__':
    time_optical_flow_path('disparity/cporta_left.png', 'disparity/cporta_right.png')
    time_optical_flow_path('disparity/office_left.png', 'disparity/office_right.png')
    time_optical_flow_path('disparity/office2_left.png', 'disparity/office2_right.png')
