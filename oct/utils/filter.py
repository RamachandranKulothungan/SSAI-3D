import numpy as np
import cv2
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

def downsample_and_resize(image, dr_h, dr_w):
    # Downsample the image
    downsampled_image = image[::dr_h, ::dr_w]
    
    # Resize the image to the original size
    resized_image = cv2.resize(downsampled_image, (image.shape[1], image.shape[0]))
    return resized_image