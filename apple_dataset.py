import os
import numpy as np
import cv2
from PIL import Image

class AppleDataset(object):
    def __init__(self, val=False, root_dir='MiniApples/detection/train'):
        self.root_dir = root_dir

        # Load all image and mask files, sorting them to ensure they are aligned
        self.img_files = list(sorted(os.listdir(os.path.join(root_dir, "images"))))
        self.mask_files = list(sorted(os.listdir(os.path.join(root_dir, "masks"))))

        if val:
            filt = lambda fn: fn.startswith('20150921_131346')
        else:
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
            img = np.array(img)
            mask = np.array(mask)
            # Each color of mask corresponds to a different instance
            # with 0 being the background
            mask[mask>0] = 1 # merge all instances
            self.imgs.append( cv2.cvtColor(img,cv2.COLOR_RGB2BGR) )
            self.masks.append( mask )

    def __getitem__(self, idx):
        return self.imgs[idx], self.masks[idx]

    def __len__(self):
        return len(self.imgs)

    def get_img_name(self, idx):
        return self.img_files[idx]
