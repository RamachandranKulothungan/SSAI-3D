import random
import tifffile
from basicsr.utils.dataset import normalize, resize
import os
import numpy as np
from oct.utils import filter
from scipy import signal
import cv2

ground_truth_dir = "gt"
low_quality_dir = "lq"
zs_ground_truth_dir = "zs_gt"
zs_low_quality_dir = "zs_lq"
projected_dir = "projected_xy"

def create_projected_data(raw_tif_pth, save_pth, project_depth = 7):

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
    return

def create_synthetic_data_from_slices(input_pth, output_pth,
    kernel_num = 3, downsample_rate = 5):
    kernel_lst = []
    res_lst = [[] for _ in range(kernel_num)]
    gt_lst = [[] for _ in range(kernel_num)]
    
    for idx, std in enumerate(np.arange(3,101,2)):
        if idx >= kernel_num:
            break
        kernel_lst.append(filter.g_filter(51, 0, std))
    
    gt_pth = os.path.join(output_pth, ground_truth_dir)
    lq_pth = os.path.join(output_pth, low_quality_dir)
    os.makedirs(gt_pth, exist_ok = True)
    os.makedirs(lq_pth, exist_ok = True)
    files = os.listdir(input_pth)

    for file in files:
        raw_slice = tifffile.imread(os.path.join(input_pth, file))
        for idx, k in enumerate(kernel_lst):
            conved_slice = signal.fftconvolve(raw_slice, k, mode = 'same')
            conved_slice = resize(conved_slice, downsample_rate, raw_slice)
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
            tifffile.imwrite(os.path.join(gt_pth, f'{idx}_{slice_idx}.tiff'), gt_slice)
            tifffile.imwrite(os.path.join(lq_pth, f'{idx}_{slice_idx}.tiff'), lq_slice)
    return

# Samples 10 images from the dataset and creates a new dataset with the samples
def create_zs_dataset(input_pth):
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
        x = lq_s.shape[0]
        y = lq_s.shape[1]

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
            slice = cv2.resize(slice, (raw_data.shape[-1]*dr, raw_data.shape[0]*dr))
            tifffile.imwrite(os.path.join(path_yz, f'{idx}.tiff'), slice)

    if xz_required:
        path_xz = os.path.join(save_pth,'test_xz')
        os.makedirs(path_xz, exist_ok=True)
        for idx in range(xz_len):
            slice = raw_data[:,:,idx]
            slice = cv2.resize(slice, (raw_data.shape[1], raw_data.shape[0]*dr))
            tifffile.imwrite(os.path.join(path_xz, f'{idx}.tiff'), slice)
