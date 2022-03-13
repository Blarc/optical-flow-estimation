import time

import cv2
import numpy as np

from horn_schunck import horn_schunck
from lucas_kanade import lucas_kanade

# ignore if division with zero
np.seterr(divide='ignore', invalid='ignore')


def time_optical_flow_path(path_1, path_2, kernel_size=10, iterations=1000):
    image_1 = cv2.imread(path_1, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    image_2 = cv2.imread(path_2, cv2.IMREAD_GRAYSCALE).astype(np.float32)

    time_optical_flow(image_1, image_2, kernel_size, iterations)


def time_optical_flow(image_1, image_2, kernel_size=10, iterations=1000):
    lk_start_time = time.process_time()
    lucas_kanade(image_1, image_2, kernel_size)
    lk_time = time.process_time() - lk_start_time

    hs_start_time = time.process_time()
    horn_schunck(image_1, image_2, iterations, 0.5)
    hs_time = time.process_time() - hs_start_time

    lk_hs_start_time = time.process_time()
    horn_schunck(image_1, image_2, iterations, 0.5, kernel_size)
    lk_hs_time = time.process_time() - lk_hs_start_time

    print('-' * 75)
    print(f'Lucas-Kanade:                      {lk_time * 1000}')
    print(f'Horn-Shunck:                       {hs_time * 1000}')
    print(f'Horn-Shunck (initalized with l-k): {lk_hs_time * 1000}')


if __name__ == '__main__':
    time_optical_flow_path('collision/00000055.jpg', 'collision/00000056.jpg', kernel_size=10, iterations=100)
    time_optical_flow_path('collision/00000055.jpg', 'collision/00000056.jpg', kernel_size=50, iterations=500)
    time_optical_flow_path('collision/00000055.jpg', 'collision/00000056.jpg', kernel_size=100, iterations=1000)
    time_optical_flow_path('lab2/001.jpg', 'lab2/002.jpg', kernel_size=10, iterations=100)
    time_optical_flow_path('lab2/001.jpg', 'lab2/002.jpg', kernel_size=50, iterations=500)
    time_optical_flow_path('lab2/001.jpg', 'lab2/002.jpg', kernel_size=100, iterations=1000)

    # ---------------------------------------------------------------------------
    # Lucas - Kanade: 15.625
    # Horn - Shunck: 1796.875
    # Horn - Shunck(initalized
    # with l - k): 953.125
    # ---------------------------------------------------------------------------
    # Lucas - Kanade: 31.25
    # Horn - Shunck: 8656.25
    # Horn - Shunck(initalized
    # with l - k): 4328.125
    # ---------------------------------------------------------------------------
    # Lucas - Kanade: 15.625
    # Horn - Shunck: 13640.625
    # Horn - Shunck(initalized
    # with l - k): 8171.875
    # ---------------------------------------------------------------------------
    # Lucas - Kanade: 15.625
    # Horn - Shunck: 4265.625
    # Horn - Shunck(initalized
    # with l - k): 1531.25
    # ---------------------------------------------------------------------------
    # Lucas - Kanade: 46.875
    # Horn - Shunck: 20218.75
    # Horn - Shunck(initalized
    # with l - k): 10359.375
    # ---------------------------------------------------------------------------
    # Lucas - Kanade: 78.125
    # Horn - Shunck: 43343.75
    # Horn - Shunck(initalized
    # with l - k): 16484.375