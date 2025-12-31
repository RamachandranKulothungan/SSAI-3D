import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
from scipy.ndimage import rotate

def g_filter(kernel_size, std_x, std_y, rotation=0):
    """
    Generate an Gaussian filter kernel.
    Parameters:
        kernel_size (int): Size of the filter kernel (assumed square).
        std_x (float): Standard deviation in the x direction.
        std_y (float): Standard deviation in the y direction.
        rotation (float): Rotation angle in degrees.
    Returns:
        ndarray: The generated Gaussian filter kernel.
    """
    # Create an impulse (delta function) image
    size = kernel_size
    impulse = np.zeros((size, size))
    impulse[size // 2, size // 2] = 1  # Center pixel is 1 (impulse)
    # Apply anisotropic Gaussian filter
    sigma_x = std_x  # Spread in x direction
    sigma_y = std_y  # Spread in y direction
    filtered = gaussian_filter(impulse, sigma=[sigma_y, sigma_x])  # Note: sigma order is (row, col)
    return rotate(filtered, angle=rotation)

def downsample_and_resize(image, dr_h, dr_w):
    """
    Downsample an image by factors dr_h and dr_w, then resize back to original size.
    Parameters:
        image (ndarray): Input image to be downsampled and resized.
        dr_h (int): Downsampling factor in height.
        dr_w (int): Downsampling factor in width.
    Returns:
        ndarray: The downsampled and resized image.
    """
    # Downsample the image
    downsampled_image = image[::dr_h, ::dr_w]
    
    # Resize the image to the original size
    resized_image = cv2.resize(downsampled_image, (image.shape[1], image.shape[0]))
    return resized_image