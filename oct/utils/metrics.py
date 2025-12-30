import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def calculate_metrics(volume_ref, volume_test, axis=0):
    """
    Calculate PSNR and SSIM slice-by-slice for OCT volumes.
    
    Parameters:
        volume_ref (ndarray): Reference OCT volume (x, y, z)
        volume_test (ndarray): Processed OCT volume of same shape
    Returns:
        psnr_list, ssim_list: lists of metrics for each slice
    """
    assert volume_ref.shape == volume_test.shape, "Volumes must have the same shape"
    
    psnr_list = []
    ssim_list = []
    
    # Iterate over slices along z-axis
    for i in range(volume_ref.shape[axis]):
        if axis==0:
            ref_slice = volume_ref[i, :, :]
            test_slice = volume_test[i, :, :]
        elif axis==1:
            ref_slice = volume_ref[:, i, :]
            test_slice = volume_test[:, i, :]
        elif axis==2:
            ref_slice = volume_ref[:, :, i]
            test_slice = volume_test[:, :, i]
        else:
            raise ValueError("Axis must be 0, 1, or 2")
        psnr_val = peak_signal_noise_ratio(ref_slice, test_slice, data_range=ref_slice.max() - ref_slice.min())
        ssim_val = structural_similarity(ref_slice, test_slice, data_range=ref_slice.max() - ref_slice.min())
        
        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)
    
    return psnr_list, ssim_list

def plot_metrics(psnr_list, ssim_list):
    slices = range(len(psnr_list))
    
    plt.figure(figsize=(12,5))
    
    # PSNR plot
    plt.subplot(1,2,1)
    plt.plot(slices, psnr_list, marker='o')
    plt.title("PSNR across slices")
    plt.xlabel("Slice index")
    plt.ylabel("PSNR (dB)")
    
    # SSIM plot
    plt.subplot(1,2,2)
    plt.plot(slices, ssim_list, marker='o', color='orange')
    plt.title("SSIM across slices")
    plt.xlabel("Slice index")
    plt.ylabel("SSIM")
    
    plt.tight_layout()
    plt.show()

