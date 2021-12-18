import os
import sys
import cv2
from final import detect_and_score
from hough_circles_trackbar import detect_and_show
from util import im_show
import tqdm as tq
import numpy as np
import torch
import torch.utils.data
from apple_dataset import AppleDataset

def collate_fn(batch):
    return tuple(zip(*batch))

def get_dataset(data_dir):
    # device = torch.device('cpu')
    dataset = AppleDataset(data_dir, None)
    return dataset

def train_all():
    dataset = get_dataset('MiniApples/detection/train')
    # data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, collate_fn=collate_fn)
    scores = []
    for img, mask in tq.tqdm(dataset):
        mask[mask>0] = 255
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # detect_and_show(img)
        s = detect_and_score(img, mask)
        scores.append(s)
    return np.array(scores).mean()

def main():
    avg_score = train_all()
    print('Average IoU = ', avg_score)

if __name__ == "__main__":
    main()
    # import cProfile, pstats
    # cProfile.run("main()", "{}.profile".format(__file__))
    # s = pstats.Stats("{}.profile".format(__file__))
    # s.strip_dirs()
    # s.sort_stats("time").print_stats(10)