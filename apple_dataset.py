import os
import numpy as np
import cv2
from PIL import Image

#####################################
# Class that takes the input instance masks
# and extracts bounding boxes on the fly
#####################################
class AppleDataset(object):
    def __init__(self, root_dir, transforms):
        self.root_dir = root_dir

        # Load all image and mask files, sorting them to ensure they are aligned
        self.img_files = list(sorted(os.listdir(os.path.join(root_dir, "images"))))
        self.mask_files = list(sorted(os.listdir(os.path.join(root_dir, "masks"))))

        filt = lambda fn: fn.startswith('20150921_131234')
        self.img_files = list(filter(filt, self.img_files))
        self.mask_files = list(filter(filt, self.mask_files))

        self.imgs = []
        self.masks = []
        for img_f, mask_f in zip(self.img_files, self.mask_files):
            img_path = os.path.join(self.root_dir, "images", img_f)
            mask_path = os.path.join(self.root_dir, "masks", mask_f)
            img = Image.open(img_path)
            mask = Image.open(mask_path)
            # Each color of mask corresponds to a different instance
            # with 0 being the background
            img = np.array(img)
            mask = np.array(mask)
            mask[mask>0] = 1
            self.imgs.append( cv2.cvtColor(img,cv2.COLOR_RGB2BGR) )
            self.masks.append( mask )

    def __getitem__(self, idx):
        return self.imgs[idx], self.masks[idx]

    def __len__(self):
        return len(self.imgs)

    def get_img_name(self, idx):
        return self.img_files[idx]
