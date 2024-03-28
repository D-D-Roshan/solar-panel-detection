

import numpy as np
from matplotlib import pyplot as plt
from patchify import patchify
import tifffile as tiff
from PIL import Image

large_image_stack = tiff.imread('C:\\Users\\Roshan D\\Desktop\\solar_panel_detection_CNN\\satellite-img-1.tif')


for img in range(large_image_stack.shape[0]):

    large_image = large_image_stack[img]
   
    
    patches_img = patchify(large_image, (512, 512), step=256)  #Step=256 for 256 patches means no overlap
    
    for i in range(patches_img.shape[0]):
        for j in range(patches_img.shape[1]):
            
            single_patch_img = patches_img[i,j,:,:]
            tiff.imwrite('patches/images/' + 'image_' + str(img) + '_' + str(i)+str(j)+ ".tif", single_patch_img)
            



