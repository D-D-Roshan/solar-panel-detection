import os
import numpy as np
from matplotlib import pyplot as plt
from patchify import patchify
import tifffile as tiff
from PIL import Image

# Create the directory for saving the TIFF files
save_dir = 'patches/images/'
os.makedirs(save_dir, exist_ok=True)

large_image_stack = tiff.imread('C:\\Users\\Roshan D\\Desktop\\solar_panel_detection_CNN\\satellite-img-1.tif')

for img in range(large_image_stack.shape[0]):
    large_image = large_image_stack[img]
    patches_img = patchify(large_image, (512, 512), step=512)

    for i in range(patches_img.shape[0]):
        for j in range(patches_img.shape[1]):
            single_patch_img = patches_img[i, j, :, :]
            filename = f'image_{img}_{i}{j}.tif'
            filepath = os.path.join(save_dir, filename)
            tiff.imwrite(filepath, single_patch_img)
