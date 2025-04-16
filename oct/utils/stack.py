import tifffile
from tqdm import tqdm
import os
import numpy as np

def create_3d_tiff(input_stack_dir, output_file):
    rec_stack = []
    dlen = len(os.listdir(input_stack_dir))

    for i in tqdm(range(dlen)):
        rec_slice = tifffile.imread(os.path.join(input_stack_dir, f'{i}.tiff'))
        rec_stack.append(rec_slice)
    print(np.stack(rec_stack).shape)
    rec_stack = np.stack(rec_stack).mean(-1)
    print(rec_stack.shape)
    rec_stack = rec_stack.astype(np.float64)
        
    tifffile.imwrite(output_file, rec_stack, bigtiff=True)