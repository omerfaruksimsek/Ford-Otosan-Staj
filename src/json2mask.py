import json
import os
import numpy as np
import cv2
import tqdm
from constant import *

augmentationJsonList = []
augmentationImageList = []
augmentationMaskList = []

# Create a list which contains every file name in "jsons" folder
json_list = os.listdir(JSON_DIR)

""" tqdm Example Start"""

iterator_example = range(1000000)

for i in tqdm.tqdm(iterator_example):
    pass

""" rqdm Example End"""


# For every json file
for json_name in tqdm.tqdm(json_list):

    # Access and open json file as dictionary
    json_path = os.path.join(JSON_DIR, json_name)
    json_file = open(json_path, 'r')

    # Load json data
    json_dict = json.load(json_file)

    # Create an empty mask whose size is the same as the original image's size
    mask = np.zeros((json_dict["size"]["height"], json_dict["size"]["width"]), dtype=np.uint8)
    mask_path = os.path.join(MASK_DIR, json_name[:-9]+".png")

    # For every objects
    for obj in json_dict["objects"]:
        # Check the objects ‘classTitle’ is ‘Freespace’ or not.
        if obj['classTitle']=='Freespace':
            # Extract exterior points which is a point list that contains
            # every edge of polygon and fill the mask with the array.
            mask = cv2.fillPoly(mask, np.array([obj['points']['exterior']]), color=1)
            if obj['points']['interior'] != []:
                #print("\n" + json_name)
                #print(len(obj['points']['interior']))
                for i in range(len(obj['points']['interior'])):
                    mask = cv2.fillPoly(mask, np.array([obj['points']['interior'][i]]), color=0)
    
    for tag in json_dict["tags"]:
        if tag['value'] == 'Tunnel':
            augmentationMask = np.zeros((json_dict["size"]["height"], json_dict["size"]["width"]), dtype=np.uint8)
            augmentationMask_path = os.path.join(MASK_DIR, json_name[:-9]+"augmentation.png")
            augmentationMask = mask
            #cv2.imwrite(augmentationMask_path, augmentationMask.astype(np.uint8))
            
            image_path = os.path.join(IMAGE_DIR, json_name[:-9]+'.jpg')
            image = cv2.imread(image_path).astype(np.uint8)
            augmentationImage  = image.copy()
            augmentationImageParth= os.path.join(IMAGE_DIR,json_name[:-9]+'augmentation.jpg')
            #cv2.imwrite(augmentationImageParth, augmentationImage)
            
            augmentationImageList.append(json_name[:-9]+'.jpg')
            augmentationJsonList.append(json_name)
            augmentationMaskList.append(json_name[:-9]+".png")

        
    # Write mask image into MASK_DIR folder
    cv2.imwrite(mask_path, mask.astype(np.uint8))
