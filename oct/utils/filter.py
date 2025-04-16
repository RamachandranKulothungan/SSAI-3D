import numpy as np
from scipy.ndimage import gaussian_filter

def g_filter(kernel_size, std_x, std_y):
    # Create an impulse (delta function) image
    size = kernel_size
    impulse = np.zeros((size, size))
    impulse[size // 2, size // 2] = 1  # Center pixel is 1 (impulse)
    # Apply anisotropic Gaussian filter
    sigma_x = std_x  # Spread in x direction
    sigma_y = std_y  # Spread in y direction
    filtered = gaussian_filter(impulse, sigma=[sigma_y, sigma_x])  # Note: sigma order is (row, col)
    return filtered