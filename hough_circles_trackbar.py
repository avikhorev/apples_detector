import cv2
from final import *
from apple_dataset import AppleDataset

def draw_circles(img, circles):
    for x,y,r in circles:
        cv2.circle(img, (x,y), r, (255,255,0), thickness=1)
        cv2.circle(img, (x,y), 2, (0,255,255), thickness=-1)
    return img

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

def detect_and_show(img):
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

    while True:
        img = img_orig

        read_all_track_bars(win_name)

        img     = blur(img)
        mask_hsv, img = color_thresholding(img)
        img = cv2.bitwise_and(img, img, mask=mask_hsv)
        gray = to_gray(img)
        circles = get_circles(gray)
        # mask_circles = get_mask(gray,circles)
        # cimg = cv2.bitwise_and(cimg, cimg, mask=mask_circles)
        gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        img2 = img_orig.copy()
        draw_circles(img2, circles)
        # draw_circles(gray, circles)
        # draw_circles(img, circles)

        img = np.concatenate([img2, gray, img], axis=1)
        cv2.imshow(win_name, img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":

    dset = AppleDataset()
    for cimg,_ in dset:
        # img_path = 'apple.jpg'
        # img_path = 'photo_2021-12-17_20-42-08.jpg'
        # cimg = cv2.imread(img_path)
        detect_and_show(cimg)
