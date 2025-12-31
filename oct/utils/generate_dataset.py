import random
import tifffile
from basicsr.utils.dataset import normalize, resize
import os
import numpy as np
from oct.utils import filter
from scipy import signal
import cv2
import matplotlib.pyplot as plt

ground_truth_dir = "gt"
low_quality_dir = "lq"
zs_ground_truth_dir = "zs_gt"
zs_low_quality_dir = "zs_lq"
projected_dir = "projected_xy"

def create_projected_data(raw_tif_pth, save_pth, project_depth = 7):
    """
    Create projected images from raw OCT data by averaging every 'project_depth' slices.
     Parameters:
        raw_tif_pth (str): Path to the raw OCT tif file.
        save_pth (str): Directory to save the projected images.
        project_depth (int): Number of slices to average for each projection.
    
    """

    raw_data = tifffile.imread(raw_tif_pth)
    raw_data = normalize(raw_data)
    # slices= number of slices when proj_idx number of images are used to create projections
    slices = int((raw_data.shape[0]//project_depth))

    projected_pth = os.path.join(save_pth, projected_dir)
    os.makedirs(projected_pth, exist_ok = True)

    # Create convolved images from projections of project_depth slice sets
    for slice_idx in range(slices):
        avg_slice = np.average(raw_data[slice_idx * project_depth:(slice_idx+1) * project_depth], 0)
        avg_slice = normalize(avg_slice)
        tifffile.imwrite(os.path.join(projected_pth, f'{slice_idx}.tiff'), avg_slice)


def create_projected_data_with_1step_slide(raw_tif_pth, save_pth, project_depth = 4, projected_dir = projected_dir):
    """
    Create projected images from raw OCT data by averaging every 'project_depth' slices with a sliding window of 1.
    Parameters:
        raw_tif_pth (str): Path to the raw OCT tif file.
        save_pth (str): Directory to save the projected images.
        project_depth (int): Number of slices to average for each projection.
    """
    raw_data = tifffile.imread(raw_tif_pth)
    raw_data = normalize(raw_data)

    projected_pth = os.path.join(save_pth, projected_dir)
    os.makedirs(projected_pth, exist_ok = True)

    # Create convolved images from projections of project_depth slice sets
    for slice_idx in range(500 -  project_depth):
        avg_slice = np.average(raw_data[slice_idx:slice_idx + project_depth], 0)
        avg_slice = normalize(avg_slice)
        tifffile.imwrite(os.path.join(projected_pth, f'{slice_idx}.tiff'), avg_slice)

def create_synthetic_data_from_slices(input_pth, output_pth,
                                    kernel_num = 3, dr_h = 2, dr_w = 2, blur_width = False, 
                                    blur_height = False, rotation=0, suffix=""):
    """
    Create synthetic low-quality OCT slices from high-quality slices by applying Gaussian blur and downsampling.
    Parameters:
        input_pth (str): Path to the directory containing high-quality OCT slices.
        output_pth (str): Directory to save the generated low-quality and ground truth slices.
        kernel_num (int): Number of different Gaussian kernels to use.
        dr_h (int): Downsampling rate in height.
        dr_w (int): Downsampling rate in width.
        blur_width (bool): Whether to apply blur in width direction.
        blur_height (bool): Whether to apply blur in height direction.
        rotation (float): Rotation angle for the Gaussian filter.
        suffix (str): Suffix to add to the saved file names.    
    """
    kernel_lst = []
    res_lst = [[] for _ in range(kernel_num)]
    gt_lst = [[] for _ in range(kernel_num)]
    
    for idx, std in enumerate(np.arange(3,101,2)):
        if idx >= kernel_num:
            break
        std_width = 0
        std_height = 0
        if blur_width:
            std_width = std
        if blur_height:
            std_height = std
        kernel_lst.append(filter.g_filter(51, std_width, std_height, rotation))
    
    for i in range(len(kernel_lst)):
        plt.imshow(kernel_lst[i], cmap='jet', extent=[-51//2, 51//2, -51//2, 51//2])
        plt.colorbar(label='Intensity')
        plt.title(f"Gaussian Filter ({i})")
        plt.show()

    gt_pth = os.path.join(output_pth, ground_truth_dir)
    lq_pth = os.path.join(output_pth, low_quality_dir)
    os.makedirs(gt_pth, exist_ok = True)
    os.makedirs(lq_pth, exist_ok = True)
    files = os.listdir(input_pth)

    for file in files:
        raw_slice = tifffile.imread(os.path.join(input_pth, file))
        for idx, k in enumerate(kernel_lst):
            conved_slice = signal.fftconvolve(raw_slice, k, mode = 'same')
            conved_slice = filter.downsample_and_resize(conved_slice, dr_h, dr_w)
            res_lst[idx].append(conved_slice)
            gt_lst[idx].append(raw_slice)
            assert conved_slice.shape == raw_slice.shape

    lqstacks = [np.stack(s) for s in res_lst]
    gtstacks = [np.stack(s) for s in gt_lst]
    for idx, stack in enumerate(lqstacks):
        lqstack = normalize(stack)
        gtstack = normalize(gtstacks[idx])
        for slice_idx, lq_slice in enumerate(lqstack):
            gt_slice = gtstack[slice_idx]
            tifffile.imwrite(os.path.join(gt_pth, f'{idx}_{slice_idx}{suffix}.tiff'), gt_slice)
            tifffile.imwrite(os.path.join(lq_pth, f'{idx}_{slice_idx}{suffix}.tiff'), lq_slice)

def create_synthetic_data_from_3d_stack(raw_tif_pth, save_pth,
                            kernel_num = 3, dr_h = 2, dr_w = 2, blur_width = False, 
                            blur_height = False, rotation=0, suffix=""):
    """
    Create synthetic low-quality OCT volumes from a high-quality volume by applying Gaussian blur and downsampling
    Parameters:
        raw_tif_pth (str): Path to the raw high-quality OCT tif file.
        save_pth (str): Directory to save the generated low-quality and ground truth volumes.
        kernel_num (int): Number of different Gaussian kernels to use.
        dr_h (int): Downsampling rate in height.
        dr_w (int): Downsampling rate in width.
        blur_width (bool): Whether to apply blur in width direction.
        blur_height (bool): Whether to apply blur in height direction.
        rotation (float): Rotation angle for the Gaussian filter.
        suffix (str): Suffix to add to the saved file names.    
    """

    raw_data = tifffile.imread(raw_tif_pth)
    raw_data = normalize(raw_data)
    kernel_lst = []
    res_lst = [[] for _ in range(kernel_num)]
    gt_lst = [[] for _ in range(kernel_num)]
    z_slices = raw_data.shape[0]
    
    for idx, std in enumerate(np.arange(3,101,2)):
        if idx >= kernel_num:
            break
        std_width = 0
        std_height = 0
        if blur_width:
            std_width = std
        if blur_height:
            std_height = std
        kernel_lst.append(filter.g_filter(51, std_width, std_height, rotation))

    for i in range(len(kernel_lst)):
        plt.imshow(kernel_lst[i], cmap='jet', extent=[-51//2, 51//2, -51//2, 51//2])
        plt.colorbar(label='Intensity')
        plt.title(f"Gaussian Filter ({i})")
        plt.show()

    gt_pth = os.path.join(save_pth, 'gt')
    lq_pth = os.path.join(save_pth, 'lq')
    os.makedirs(gt_pth, exist_ok = True)
    os.makedirs(lq_pth, exist_ok = True)
    for slice_idx in range(z_slices):
        raw_slice = raw_data[slice_idx]
        for idx, k in enumerate(kernel_lst):
            conved_slice = signal.fftconvolve(raw_slice, k, mode = 'same')
            conved_slice = filter.downsample_and_resize(conved_slice, dr_h, dr_w)
            res_lst[idx].append(conved_slice)
            gt_lst[idx].append(raw_slice)
            assert conved_slice.shape == raw_slice.shape

    lqstacks = [np.stack(s) for s in res_lst]
    gtstacks = [np.stack(s) for s in gt_lst]
    for idx, stack in enumerate(lqstacks):
        lqstack = normalize(stack)
        gtstack = normalize(gtstacks[idx])
        for slice_idx, lq_slice in enumerate(lqstack):
            gt_slice = gtstack[slice_idx]
            tifffile.imwrite(os.path.join(gt_pth, f'{idx}_{slice_idx}{suffix}.tiff'), gt_slice)
            tifffile.imwrite(os.path.join(lq_pth, f'{idx}_{slice_idx}{suffix}.tiff'), lq_slice)

def create_downsampled_data_from_3d_stack(raw_tif_pth, output_pth, dr_h = 2, dr_w = 2):
    """
    Create downsampled low-quality OCT volume from a high-quality volume.
    Parameters:
        raw_tif_pth (str): Path to the raw high-quality OCT tif file.
        output_pth (str): Directory to save the generated low-quality and ground truth volumes.
        dr_h (int): Downsampling rate in height.
        dr_w (int): Downsampling rate in width.
    """
    raw_data = tifffile.imread(raw_tif_pth)
    raw_data = normalize(raw_data)
    res_lst = []
    gt_lst = []
    z_slices = raw_data.shape[0]
    
    gt_pth = os.path.join(output_pth, ground_truth_dir)
    lq_pth = os.path.join(output_pth, low_quality_dir)
    os.makedirs(gt_pth, exist_ok = True)
    os.makedirs(lq_pth, exist_ok = True)

    for slice_idx in range(z_slices):
        raw_slice = raw_data[slice_idx]
        dr_slice = filter.downsample_and_resize(raw_slice, dr_h, dr_w)
        res_lst.append(dr_slice)
        gt_lst.append(raw_slice)
        assert dr_slice.shape == raw_slice.shape

    lqstack = np.stack(res_lst)
    gtstack = np.stack(gt_lst)
    for slice_idx, lq_slice in enumerate(lqstack):
        gt_slice = gtstack[slice_idx]
        lq_slice = normalize(lq_slice)
        gt_slice = normalize(gt_slice)
        tifffile.imwrite(os.path.join(gt_pth, f'dr_{slice_idx}.tiff'), gt_slice)
        tifffile.imwrite(os.path.join(lq_pth, f'dr_{slice_idx}.tiff'), lq_slice)

def create_downsampled_data_from_slices(input_pth, output_pth, dr_h = 2, dr_w = 2):
    """
    Create downsampled low-quality OCT slices from high-quality slices.
    Parameters:
        input_pth (str): Path to the directory containing high-quality OCT slices.
        output_pth (str): Directory to save the generated low-quality and ground truth slices.
        dr_h (int): Downsampling rate in height.
        dr_w (int): Downsampling rate in width.
    """
    res_lst = []
    gt_lst = []

    gt_pth = os.path.join(output_pth, ground_truth_dir)
    lq_pth = os.path.join(output_pth, low_quality_dir)
    os.makedirs(gt_pth, exist_ok = True)
    os.makedirs(lq_pth, exist_ok = True)
    files = os.listdir(input_pth)

    for file in files:
        raw_slice = tifffile.imread(os.path.join(input_pth, file))
        dr_slice = filter.downsample_and_resize(raw_slice, dr_h, dr_w)
        res_lst.append(dr_slice)
        gt_lst.append(raw_slice)
        assert dr_slice.shape == raw_slice.shape

    lqstack = np.stack(res_lst)
    gtstack = np.stack(gt_lst)
    for slice_idx, lq_slice in enumerate(lqstack):
        gt_slice = gtstack[slice_idx]
        lq_slice = normalize(lq_slice)
        gt_slice = normalize(gt_slice)
        tifffile.imwrite(os.path.join(gt_pth, f'dr_{slice_idx}.tiff'), gt_slice)
        tifffile.imwrite(os.path.join(lq_pth, f'dr_{slice_idx}.tiff'), lq_slice)

# Samples 10 images from the dataset and creates a new dataset with the samples
def create_zs_dataset(input_pth):
    """
    Create a zero-shot dataset by sampling 10 random images from the existing dataset.
    Snapshots are saved in 'zs_gt' and 'zs_lq' directories.
    Takes small crops from the center of the images for faster processing.

    Parameters:
        input_pth (str): Path to the directory containing high-quality and low-quality OCT slices
    """
    os.makedirs(os.path.join(input_pth, zs_ground_truth_dir), exist_ok = True)
    os.makedirs(os.path.join(input_pth, zs_low_quality_dir), exist_ok = True)
    gt_pth = os.path.join(input_pth, ground_truth_dir)
    lq_pth = os.path.join(input_pth, low_quality_dir)
    file_names = os.listdir(gt_pth)

    d = 100
    files = random.sample(file_names, 10)
    for file in files:
        gt_s = tifffile.imread(os.path.join(gt_pth, file))
        lq_s = tifffile.imread(os.path.join(lq_pth, file))

        x_center, y_center = lq_s.shape[0] // 2, lq_s.shape[1] // 2
        gt_s = gt_s[x_center-2*d:x_center+d, y_center-2*d:y_center+d]
        lq_s = lq_s[x_center-2*d:x_center+d, y_center-2*d:y_center+d]

        gt_s = gt_s/gt_s.max()
        gt_s = gt_s * 255
        gt_s = gt_s.astype(np.uint8)

        lq_s = lq_s/lq_s.max()
        lq_s = lq_s * 255
        lq_s = lq_s.astype(np.uint8)
        tifffile.imwrite(os.path.join(input_pth, zs_low_quality_dir, file), lq_s)
        tifffile.imwrite(os.path.join(input_pth, zs_ground_truth_dir, file), gt_s)

def generate_oct_raw_data(raw_pth, save_pth, dr, xy_required=False, xz_required=False, yz_required=False):
    """
    Generate OCT raw data slices in specified planes from a 3D raw OCT volume.
    
    Parameters:
        raw_pth (str): Path to the raw OCT tif file.
        save_pth (str): Path to the directory where the generated slices will be saved.
        dr (int): Downsampling rate.
        xy_required (bool): Whether to generate XY slices.
        xz_required (bool): Whether to generate XZ slices.
        yz_required (bool): Whether to generate YZ slices.
    """
    raw_data = tifffile.imread(raw_pth)
    raw_data = normalize(raw_data)
    print(raw_data.dtype)
    print(raw_data.shape)
    
    assert len(raw_data.shape) == 3
    xz_len = raw_data.shape[-1]
    yz_len = raw_data.shape[1]
    xy_len = raw_data.shape[0]

    if xy_required:
        path_xy = os.path.join(save_pth,'test_xy')    
        os.makedirs(path_xy, exist_ok=True)
        for idx in range(xy_len):
            slice = raw_data[idx]
            slice = cv2.resize(slice, (raw_data.shape[-1]*dr, raw_data.shape[1]))
            tifffile.imwrite(os.path.join(path_xy, f'{idx}.tiff'), slice)

    if yz_required:
        path_yz = os.path.join(save_pth,'test_yz')
        os.makedirs(path_yz, exist_ok=True)
        for idx in range(yz_len):
            slice = raw_data[:,idx]
            slice = cv2.resize(slice, (raw_data.shape[-1], raw_data.shape[0]*dr))
            tifffile.imwrite(os.path.join(path_yz, f'{idx}.tiff'), slice)

    if xz_required:
        path_xz = os.path.join(save_pth,'test_xz')
        os.makedirs(path_xz, exist_ok=True)
        for idx in range(xz_len):
            slice = raw_data[:,:,idx]
            slice = cv2.resize(slice, (raw_data.shape[1], raw_data.shape[0]*dr))
            tifffile.imwrite(os.path.join(path_xz, f'{idx}.tiff'), slice)

