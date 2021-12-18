import os
import sys
import cv2
from util import im_show
import tqdm as tq
import numpy as np
import torch
import torch.utils.data
from apple_dataset import AppleDataset
from hough_circles_trackbar import detect_and_score


# def get_transform(train):
#     transforms = []
#     transforms.append(T.ToTensor())
#     if train:
#         transforms.append(T.RandomHorizontalFlip(0.5))
#     return T.Compose(transforms)

def collate_fn(batch):
    return tuple(zip(*batch))

def train_all():
    device = torch.device('cpu')
    data_dir = 'MiniApples/detection/train'

    dataset = AppleDataset(data_dir, None)
    # data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, collate_fn=collate_fn)
    scores = []
    for image, mask in tq.tqdm(dataset):
        mask[mask>0] = 255
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        s = detect_and_score(image, mask)
        scores.append(s)

        # im_show(image)
        # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        # im_id = targets[0]['image_id'].item()
        # im_name = data_loader.dataset.get_img_name(im_id)
    return np.array(scores).mean()
