import os
import cv2
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from constant import *
from json2mask import *
from torchvision import transforms as T
from PIL import Image
from PIL import ImageOps


for image in tqdm.tqdm(augmentationImageList):
    
    image_path = os.path.join(IMAGE_DIR, image)
    
    
    img=Image.open(image_path)
    augmentationImage  = img.copy()
    
    color_aug = T.ColorJitter(brightness=0.2, contrast=0.1, hue=0.006)
    augmentationImage = color_aug(augmentationImage)
    augmentationImage=np.array(augmentationImage)
    
    
    augmentationImageParth= os.path.join(AUG_IMG_DIR,image.split('.')[0]+'augmentation1.jpg')
    cv2.imwrite(augmentationImageParth,augmentationImage)    
    
    
    
    mask_path = os.path.join(MASK_DIR, image.split('.')[0]+'.png')
    mask  = cv2.imread(mask_path, 0).astype(np.uint8)
    augmentationMask  = mask.copy()
    augmentationMask_path = os.path.join(AUG_MASK_DIR, image.split('.')[0]+'augmentation1.png')
    cv2.imwrite(augmentationMask_path, augmentationMask.astype(np.uint8))
    


for image in tqdm.tqdm(augmentationImageList):
    
    image_path = os.path.join(IMAGE_DIR, image)
    
    img = cv2.imread(image_path).astype(np.uint8)
    augmentationImage  = img.copy()

    augmentationImage=np.flip(augmentationImage,1) 
    
  
    augmentationImageParth= os.path.join(AUG_IMG_DIR,image.split('.')[0]+'augmentation2.jpg')
    cv2.imwrite(augmentationImageParth,augmentationImage)    
 


    mask_path = os.path.join(MASK_DIR, image.split('.')[0]+'.png')
    mask  = cv2.imread(mask_path, 0).astype(np.uint8)
    augmentationMask  = mask.copy()
    #augmentationMask=np.array(augmentationMask)
    augmentationMask=np.flip(augmentationMask,1)
    #augmentationMask=np.array(augmentationMask)
    augmentationMask_path = os.path.join(AUG_MASK_DIR, image.split('.')[0]+'augmentation2.png')
    cv2.imwrite(augmentationMask_path, augmentationMask.astype(np.uint8))

    
    
    
    
    