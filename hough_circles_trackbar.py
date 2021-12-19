import cv2
from final import *
from apple_dataset import AppleDataset

def draw_circles(img, circles):
    for x,y,r in circles:
        cv2.circle(img, (x,y), r, (255,255,0), thickness=1)
        cv2.circle(img, (x,y), 2, (0,255,255), thickness=-1)

def draw_text_in_corner(img, text):
    cv2.putText(img, text, (10,img.shape[0]-10), cv2.FONT_HERSHEY_COMPLEX, 1.2, (0,255,255), 2)

def create_track_bar(param_name, win_name, min_val, max_val):
    nothing = lambda x: None
    cv2.createTrackbar(param_name, win_name, min_val, max_val, nothing)
    if type(PAR[param_name])==bool:
        par_value = int(PAR[param_name])
    elif type(PAR[param_name])==float:
        par_value = int(PAR[param_name]*100)
    else:
        par_value = PAR[param_name]
    cv2.setTrackbarPos(param_name, win_name, par_value)

def read_all_track_bars(win_name):
    for param_name in PAR:
        if type(PAR[param_name])==bool:
            PAR[param_name] = bool(cv2.getTrackbarPos(param_name, win_name))
        elif type(PAR[param_name])==float:
            PAR[param_name] = 1e-6 + cv2.getTrackbarPos(param_name, win_name)/100.
        else:
            PAR[param_name] = cv2.getTrackbarPos(param_name, win_name)

def detect_and_show(img, gt_mask):
    img_orig = img
    win_name = 'image'
    cv2.namedWindow( win_name, cv2.WINDOW_NORMAL )
    cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Hue is from 0-179 for Opencv
    create_track_bar('hMin', win_name, 0, 179)
    create_track_bar('hMax', win_name, 0, 179)

    create_track_bar('sMin', win_name, 0, 255)
    create_track_bar('sMax', win_name, 0, 255)

    create_track_bar('vMin', win_name, 0, 255)
    create_track_bar('vMax', win_name, 0, 255)

    create_track_bar('blur', win_name, 1, 30)

    create_track_bar('dp', win_name, 10, 1000)
    create_track_bar('param1', win_name, 1, 500)
    create_track_bar('param2', win_name, 1, 100)

    create_track_bar('min_dist', win_name, 1, 1000)
    create_track_bar('radius', win_name, 1, 100)

    create_track_bar('binarize', win_name, 0, 1)
    create_track_bar('equalize_hist', win_name, 0, 1)

    img_sep = np.full( (img.shape[0],1,3), 255, dtype=np.uint8 )
    gt_mask_orig = gt_mask.copy()
    gt_mask[gt_mask>0]=255
    gt_mask = cv2.cvtColor(gt_mask, cv2.COLOR_GRAY2BGR)

    while True:
        img = img_orig

        read_all_track_bars(win_name)

        img     = blur(img)
        mask_hsv, img = color_thresholding(img)
        img = cv2.bitwise_and(img, img, mask=mask_hsv)
        gray = to_gray(img)
        circles = get_circles(gray)
        mask_circles = get_mask(gray,circles)
        IoU = get_score(mask_circles, gt_mask_orig)
        mask_circles[mask_circles>0]=255
        mask_circles = cv2.cvtColor(mask_circles, cv2.COLOR_GRAY2BGR)

        img1 = img_orig.copy()
        img2 = img_orig.copy()
        draw_circles(img2, circles)

        draw_text_in_corner(img1, 'Original')
        draw_text_in_corner(img, 'Color threshold')
        draw_text_in_corner(img2, 'Detected circles')
        draw_text_in_corner(mask_circles, f'Segmentation mask, IoU={IoU:.3f}')
        draw_text_in_corner(gt_mask, 'Ground truth mask')

        img = np.concatenate(
            [img1, img_sep, img, img_sep, img2,img_sep, mask_circles, img_sep, gt_mask],
            axis=1
        )
        cv2.imshow(win_name, img)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            return True
        if key & 0xFF == ord(' '):
            break

    return False

if __name__ == "__main__":

    dset = AppleDataset()
    for cimg,gt_mask in dset:
        # img_path = 'apple.jpg'
        # img_path = 'photo_2021-12-17_20-42-08.jpg'
        # cimg = cv2.imread(img_path)

        done = detect_and_show(cimg, gt_mask)
        cv2.destroyAllWindows()
        if done: break

