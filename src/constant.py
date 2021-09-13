import os

# Path to original images
IMAGE_DIR = '..\\data\\images\\'

# Path to jsons
JSON_DIR = '..\\data\\jsons\\'

# Path to mask
MASK_DIR  = '..\\data\\masks\\'
if not os.path.exists(MASK_DIR):
    os.mkdir(MASK_DIR)

# Path to output images
IMAGE_OUT_DIR = '..\\data\\masked_images\\'
if not os.path.exists(IMAGE_OUT_DIR):
    os.mkdir(IMAGE_OUT_DIR)

PREDİCTS_DIR  = '..\\data\\predicts\\'
if not os.path.exists(PREDİCTS_DIR):
    os.mkdir(PREDİCTS_DIR)

AUG_IMG_DIR = '..\\data\\augmentation\\images\\'
if not os.path.exists(IMAGE_OUT_DIR):
    os.mkdir(AUG_IMG_DIR)

AUG_MASK_DIR = '..\\data\\augmentation\\masks\\'
if not os.path.exists(IMAGE_OUT_DIR):
    os.mkdir(AUG_MASK_DIR)
    
AUG_IMAGE_OUT_DIR = '..\\data\\augmentation\\masked_images\\'
if not os.path.exists(IMAGE_OUT_DIR):
    os.mkdir(AUG_IMAGE_OUT_DIR)
    



# In order to visualize masked-image(s), change "False" with "True"
VISUALIZE = False

# Bacth size
BACTH_SIZE = 4

# Input dimension
HEIGHT = 224
WIDTH = 224

# Number of class, for this task it is 2: Non-drivable area and Driviable area
N_CLASS= 2