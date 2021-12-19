##################################################################
# Training and evaluation module
##################################################################


from final import detect_and_score
import tqdm as tq
import numpy as np
from apple_dataset import AppleDataset
# from hough_circles_trackbar import detect_and_show

def collate_fn(batch):
    return tuple(zip(*batch))

class Evaluator:
    def __init__(self):
        self.dataset = AppleDataset()

    def eval_model(self):
        # data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, collate_fn=collate_fn)
        scores = []
        for img, mask in tq.tqdm(self.dataset):
            # detect_and_show(img)
            s = detect_and_score(img, mask)
            scores.append(s)
        return np.array(scores).mean()

def main():
    ev = Evaluator()
    avg_score = ev.eval_model()
    print('Average IoU = ', avg_score)

def profile():
    import cProfile, pstats
    cProfile.run("main()", "{}.profile".format(__file__))
    s = pstats.Stats("{}.profile".format(__file__))
    s.strip_dirs()
    s.sort_stats("time").print_stats(10)

if __name__ == "__main__":
    main()
    #profile()
