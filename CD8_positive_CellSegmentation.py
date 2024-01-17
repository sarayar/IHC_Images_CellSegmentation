# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 10:36:33 2024

@author: SARABYA
"""


import numpy as np
import matplotlib.pyplot as plt

from skimage import data, io, img_as_ubyte
from skimage.color import rgb2hed, hed2rgb
from skimage.exposure import rescale_intensity
import cv2

# Separate the individual stains from the IHC image
def color_separate(ihc_rgb):

    #Convert the RGB image to HED using the prebuilt skimage method
    ihc_hed = rgb2hed(ihc_rgb)
    
    # Create an RGB image for each of the separated stains
    #Convert them to ubyte for easy saving to drive as an image
    null = np.zeros_like(ihc_hed[:, :, 0])
    ihc_h = img_as_ubyte(hed2rgb(np.stack((ihc_hed[:, :, 0], null, null), axis=-1)))
    ihc_e = img_as_ubyte(hed2rgb(np.stack((null, ihc_hed[:, :, 1], null), axis=-1)))
    ihc_d = img_as_ubyte(hed2rgb(np.stack((null, null, ihc_hed[:, :, 2]), axis=-1)))

    #Optional fun exercise of combining H and DAB stains into a single image with fluorescence look
    
    h = rescale_intensity(ihc_hed[:, :, 0], out_range=(0, 1),
                          in_range=(0, np.percentile(ihc_hed[:, :, 0], 99)))
    d = rescale_intensity(ihc_hed[:, :, 2], out_range=(0, 1),
                          in_range=(0, np.percentile(ihc_hed[:, :, 2], 99)))

# Cast the two channels into an RGB image, as the blue and green channels
#Convert to ubyte for easy saving as image to local drive
    zdh = img_as_ubyte(np.dstack((null, d, h))) #DAB in green and H in Blue

    return (ihc_h, ihc_e, ihc_d, zdh)

def get_tile_name (filename):
    S = filename.split('/')[4]
    tile_name = S.split('.')[0]
    return tile_name

filename = "C:/CCIPD_atEmory/Ovarian/tiles/37_13_original.tif"
tile_name = get_tile_name(filename)
ihc_rgb =io.imread(filename) 

filename = "C:/CCIPD_atEmory/Ovarian/tiles/4_17_original.tif"
tile_name = get_tile_name(filename)
ihc_rgb =io.imread(filename) 

filename = "C:/CCIPD_atEmory/Ovarian/tiles/4_24_original.tif"
tile_name = get_tile_name(filename)
ihc_rgb =io.imread(filename)

filename = "C:/CCIPD_atEmory/Ovarian/tiles/8_14_original.tif"
tile_name = get_tile_name(filename)
ihc_rgb =io.imread(filename)

filename = "C:/CCIPD_atEmory/Ovarian/tiles/13_29_original.tif"
tile_name = get_tile_name(filename)
ihc_rgb =io.imread(filename)



plt.imshow(ihc_rgb)
plt.axis("off")

H,E,D,HD = color_separate(ihc_rgb)
#plt.imsave('C:/CCIPD_atEmory/Ovarian/H_img.jpg', H)
#plt.imsave('C:/CCIPD_atEmory/Ovarian/DAB_img.jpg', D)

plt.imshow(D)
plt.axis("off")

################################################
#Segmentation using Voronoi-Otsu labeling
# For installation instructions of the package, please refer to the following link
# https://github.com/clEsperanto/pyclesperanto_prototype
##########################


import pyclesperanto_prototype as cle

# select a specific OpenCL / GPU device and see which one was chosen
device = cle.select_device('RTX')
print("Used GPU: ", device)

input_image = np.invert(D[:,:,2])
plt.imshow(input_image, cmap='gray')
#Before segmenting the image, need to push it to GPU memory. For visualisation purposes we crop out a sub-region:
input_gpu = cle.push(input_image)


cle.imshow(input_gpu)
    
sigma_spot_detection = 8 # 10
sigma_outline = 5

segmented = cle.voronoi_otsu_labeling(input_gpu, spot_sigma=sigma_spot_detection, 
                                      outline_sigma=sigma_outline)

cle.imshow(segmented, labels=True)

plt.imsave('C:/CCIPD_atEmory/Ovarian/mask.jpg', segmented)


#mask_file =io.imread("C:/CCIPD_atEmory/Ovarian/mask.jpg") 
#mask_file2 = cv2.imread('C:/CCIPD_atEmory/Ovarian/mask.jpg', cv2.IMREAD_GRAYSCALE)



############ Binarize RGB
mask_file2 = cv2.imread('C:/CCIPD_atEmory/Ovarian/mask.jpg')
# convert the input image to grayscale
gray = cv2.cvtColor(mask_file2, cv2.COLOR_BGR2GRAY)

# apply thresholding to convert grayscale to binary image
ret,thresh = cv2.threshold(gray,36,255,0) # (gray,50,255,0)

# Display the Binary Image
cv2.imshow("Binary Image", thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()

plt.imsave('C:/CCIPD_atEmory/Ovarian/binary_mask.jpg', thresh)

################## remove small components that are not cells - Saving the last version of masks
def bwareaopen(img, min_size, connectivity=8):
        """Remove small objects from binary image (approximation of 
        bwareaopen in Matlab for 2D images).
    
        Args:
            img: a binary image (dtype=uint8) to remove small objects from
            min_size: minimum size (in pixels) for an object to remain in the image
            connectivity: Pixel connectivity; either 4 (connected via edges) or 8 (connected via edges and corners).
    
        Returns:
            the binary image with small objects removed
        """
        # Find all connected components (called here "labels")
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            img, connectivity=connectivity)
        
        # check size of all connected components (area in pixels)
        for i in range(num_labels):
            label_size = stats[i, cv2.CC_STAT_AREA]
            
            # remove connected components smaller than min_size
            if label_size < min_size:
                img[labels == i] = 0
                
        return img

plt.imshow(thresh)
cleaned = thresh # to avoid changes in thresh
bwareaopen(cleaned,600,8) # This function cleans and changes the input image
plt.imshow(cleaned)
save_path = "C:/CCIPD_atEmory/Ovarian/results/" 
output_name = save_path + tile_name + '_cleaned_mask.png'
plt.imsave(output_name, cleaned)

############################# Overlay 1 #################################
where_1 = np.where(cleaned >0)

tile = ihc_rgb

tile
tile[where_1] = 0
plt.imshow(tile)

############################# Overlay 2 #################################

# read image
img = cv2.imread('C:/CCIPD_atEmory/Ovarian/tiles/13_29_original.tif')

# Assuming "cleaned" is your binary mask
mask = cv2.imread('C:/CCIPD_atEmory/Ovarian/results/13_29_original_cleaned_mask.png', cv2.IMREAD_GRAYSCALE)
_, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)

# create cyan image
cyan = np.full_like(img,(255,255,0))

# add cyan to img and save as new image
blend = 0.5
img_cyan = cv2.addWeighted(img, blend, cyan, 1-blend, 0)

# combine img and img_cyan using mask
result = np.where(mask[:, :, None] > 0, img_cyan, img)

plt.imshow(result)

cv2.namedWindow('custom window', cv2.WINDOW_NORMAL)
cv2.imshow('custom window', result)
cv2.resizeWindow('custom window', 600, 600)  # Adjust the size as needed
cv2.waitKey(0)
cv2.destroyAllWindows()
################################################################


statistics = cle.statistics_of_labelled_pixels(input_gpu, segmented) 

import pandas as pd
table = pd.DataFrame(statistics)    

print(table.describe())
print(table.info())
