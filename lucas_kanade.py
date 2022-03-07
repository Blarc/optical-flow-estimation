from ex1_utils import calculate_derivatives


def lucas_kanade(img1, img2, N):
    """
    img1 - first image matrix (grayscale)
    img2 - second image matrix (grayscale)
    n   - size of the neighbourhood (N x N)
    """
    i_x, i_y, i_t = calculate_derivatives(img1, img2, 1, 0.4)
    return i_x, i_y

