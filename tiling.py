# -*- coding: utf-8 -*-


# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 10:48:28 2023

@author: SARABYA
"""

from openslide import open_slide
import openslide
from PIL import Image 
import numpy as np
from matplotlib import pyplot as plt
import tifffile as tiff



slide=open_slide("C:\\CCIPD_atEmory\\Ovarian\\images\\2034MOT_PT.svs")

#############################################

slide_props=slide.properties
print(slide_props)


print("Vendor is:", slide_props['openslide.vendor'])
print("Pixel size of X in um is:", slide_props['openslide.mpp-x'])
print("Pixel size of Y in um is:", slide_props['openslide.mpp-y'])

#Objective used to capture the image
objective = float(slide.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER])
print("The objective power is: ", objective)

# get slide dimensions for the level 0 - max resolution level
slide_dims = slide.dimensions
print(slide_dims)
# 2nd method for getting slide dimensions for the level 0 - max resolution level
print(slide.level_dimensions[0])

#Get a thumbnail of the image and visualize
slide_thumb_600 = slide.get_thumbnail(size=(600, 600))
slide_thumb_600.show()

#Convert thumbnail (or any other PIL Image) to numpy array
slide_thumb_600_np = np.array(slide_thumb_600)
plt.figure(figsize=(8,8))
plt.imshow(slide_thumb_600_np)  

#############################################

from openslide.deepzoom import DeepZoomGenerator

tiles = DeepZoomGenerator(slide, tile_size=1024, overlap=0, limit_bounds=False)
#Here, we have divided our svs into tiles of size 1024 with no overlap. 

#The tiles object also contains data at many levels. 
#To check the number of levels
print("The number of levels in the tiles object are: ", tiles.level_count)

nl=tiles.level_count

#Generate object for tiles using the DeepZoomGenerator
tiles = DeepZoomGenerator(slide, tile_size=2048, overlap=0, limit_bounds=False)
#Here, we have divided our svs into tiles of size 256 with no overlap. 

#The tiles object also contains data at many levels. 
#To check the number of levels
print("The number of levels in the tiles object are: ", tiles.level_count)
print("The dimensions of data in each level are: ", tiles.level_dimensions)
#Total number of tiles in the tiles object
print("Total number of tiles = : ", tiles.tile_count)

###### processing and saving each tile to local directory
cols, rows = tiles.level_tiles[nl-1]


orig_tile_dir_name = "C:/CCIPD_atEmory/Ovarian/tiles/"


for row in range(rows):
    for col in range(cols):
        tile_name = str(col) + "_" + str(row)
        #tile_name = os.path.join(tile_dir, '%d_%d' % (col, row))
        #print("Now processing tile with title: ", tile_name)
        temp_tile = tiles.get_tile(nl-1, (col, row))
        temp_tile_RGB = temp_tile.convert('RGB')
        temp_tile_np = np.array(temp_tile_RGB)
        #Save original tile
        #tiff.imsave(orig_tile_dir_name+tile_name + "_original.tif", temp_tile_np)
        
        if temp_tile_np.mean() < 230 and temp_tile_np.std() > 15:
            print("Processing tile number:", tile_name)
            tiff.imsave(orig_tile_dir_name+tile_name + "_original.tif", temp_tile_np)
            #norm_img, H_img, E_img = norm_HnE(temp_tile_np, Io=240, alpha=1, beta=0.15)

            
        else:
            print("NOT PROCESSING TILE:", tile_name)
        
        
        




