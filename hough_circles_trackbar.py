import cv2
from final import *


def detect_and_show(img):
    img_orig = img
    win_name = 'image'
    cv2.namedWindow( win_name, cv2.WINDOW_NORMAL | cv2.WINDOW_FULLSCREEN)

    nothing = lambda x: None

    # Hue is from 0-179 for Opencv
    cv2.createTrackbar('hMin', win_name, 0, 179, nothing)
    cv2.createTrackbar('hMax', win_name, 0, 179, nothing)

    cv2.createTrackbar('sMin', win_name, 0, 255, nothing)
    cv2.createTrackbar('sMax', win_name, 0, 255, nothing)

    cv2.createTrackbar('VMin', win_name, 0, 255, nothing)
    cv2.createTrackbar('vMax', win_name, 0, 255, nothing)

    cv2.createTrackbar('blur', win_name, 1, 30, nothing)
    cv2.createTrackbar('param1', win_name, 1, 500, nothing)
    cv2.createTrackbar('param2', win_name, 1, 100, nothing)
    cv2.createTrackbar('min_dist', win_name, 1, 1000, nothing)
    cv2.createTrackbar('radius', win_name, 1, 100, nothing)

    cv2.setTrackbarPos('hMin', win_name, PAR['hMin'])
    cv2.setTrackbarPos('hMax', win_name, PAR['hMax'])
    cv2.setTrackbarPos('sMin', win_name, PAR['sMin'])
    cv2.setTrackbarPos('sMax', win_name, PAR['sMax'])
    cv2.setTrackbarPos('vMin', win_name, PAR['vMin'])
    cv2.setTrackbarPos('vMax', win_name, PAR['vMax'])

    cv2.setTrackbarPos('blur', win_name, PAR['blur'])

    cv2.setTrackbarPos('param1',   win_name, PAR['param1'])
    cv2.setTrackbarPos('param2',   win_name, PAR['param2'])
    cv2.setTrackbarPos('min_dist', win_name, PAR['min_dist'])
    cv2.setTrackbarPos('radius',   win_name, PAR['radius'])

    while True:
        img = img_orig

        PAR['hMin'] = cv2.getTrackbarPos('hMin', win_name)
        PAR['hMax'] = cv2.getTrackbarPos('hMax', win_name)
        PAR['sMin'] = cv2.getTrackbarPos('sMin', win_name)
        PAR['sMax'] = cv2.getTrackbarPos('sMax', win_name)
        PAR['vMin'] = cv2.getTrackbarPos('VMin', win_name)
        PAR['vMax'] = cv2.getTrackbarPos('vMax', win_name)

        PAR['blur'] = cv2.getTrackbarPos('blur', win_name)

        PAR['dp'] = 2
        PAR['param1'] = cv2.getTrackbarPos('param1', win_name)
        PAR['param2'] = cv2.getTrackbarPos('param2', win_name)
        PAR['min_dist'] = cv2.getTrackbarPos('min_dist', win_name)
        PAR['radius'] = cv2.getTrackbarPos('radius', win_name)

        img     = blur(img)
        mask_hsv = color_thresholding(img)
        img = cv2.bitwise_and(img, img, mask=mask_hsv)
        gray = to_gray(img)
        circles = get_circles(gray)
        # mask_circles = get_mask(gray,circles)
        # cimg = cv2.bitwise_and(cimg, cimg, mask=mask_circles)
        for x,y,r in circles:
            cv2.circle(img, (x,y), r, (255,255,0), thickness=2)
            cv2.circle(img, (x,y), 2, (0,255,255), thickness=-1)

        cv2.imshow(win_name, cv2.resize(img,[1200,600]))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    img_path = 'apple.jpg'
    # img_path = 'photo_2021-12-17_20-42-08.jpg'

    cimg_orig = cv2.imread(img_path)
    detect_and_show(cimg_orig)
