import cv2
import numpy as np

PAR = {'blur': 19, 'hMin': 0, 'hMax': 255, 'sMin': 0, 'sMax': 255, 'vMin': 0, 'vMax': 255, 'binarize': False, 'equalize_hist': False, 'dp': 2.49615668330778, 'param1': 11, 'param2': 25, 'min_dist': 3, 'radius': 11}

def get_score(mask1, mask2):
    I = np.bitwise_and(mask1,mask2).sum()
    U = np.bitwise_or(mask1,mask2).sum()
    IoU = I/U
    # score = np.bitwise_xor(mask1,mask2).sum()
    return IoU

def get_mask(img, circles):
    mask = np.zeros_like(img)
    if circles is None:
        return mask
    for x, y, r in circles:
        cv2.circle(mask, (x,y), r, color=1, thickness=-1)
    return mask

def get_circles(gray_img):
    dp = PAR['dp']
    param1 = PAR['param1']+1
    param2 = PAR['param2']
    min_dist = PAR['min_dist']+1
    min_radius = PAR['radius']
    max_radius = PAR['radius']+10
    circles = cv2.HoughCircles(
                        gray_img,
                        cv2.HOUGH_GRADIENT,
                        dp=dp,   # accumulator resolution (image_size/dp)
                        minDist=min_dist,  # minimum distance between two circles
                        param1=param1, # canny high threshold
                        param2=param2, # minimum number of votes
                        minRadius=min_radius,
                        maxRadius=max_radius
    )
    if circles is None:
        return []
    circles = np.uint16(circles[0,:])
    return filter(lambda c: len(c.shape)>0, circles)

def blur(cimg):
    ker_sz = PAR['blur']
    if ker_sz%2==0: ker_sz+=1
    return cv2.GaussianBlur(cimg, (ker_sz,ker_sz), 0)

def color_thresholding(cimg):
    #cimg[..., ::-1]
    hsv = cv2.cvtColor(cimg, cv2.COLOR_BGR2HSV)

    if PAR['equalize_hist']:
        hsv[:,:,2] = cv2.equalizeHist(hsv[:,:,0])
        cimg = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    hMin = PAR['hMin']
    hMax = PAR['hMax']
    sMin = PAR['sMin']
    sMax = PAR['sMax']
    vMin = PAR['vMin']
    vMax = PAR['vMax']

    h,s,v = cv2.split(hsv)
    mask_hsv = (h<=hMax)|(h>=hMin) if hMin>hMax else (hMin<=h)&(h<=hMax)
    mask_hsv &= (sMin<=s)&(s<=sMax) & (vMin<=v)&(s<=vMax)
    mask_hsv = mask_hsv.astype(np.uint8)
    return mask_hsv

def to_gray(cimg):
    gray = cv2.cvtColor(cimg, cv2.COLOR_BGR2GRAY)
    # result[result==0] = 255
    if PAR['binarize']:
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)[1]
    return gray

def detect_apples(cimg):
    cimg     = blur(cimg)
    mask_hsv = color_thresholding(cimg)
    cimg = cv2.bitwise_and(cimg, cimg, mask=mask_hsv)
    gray = to_gray(cimg)
    circles = get_circles(gray)
    mask_circles = get_mask(gray,circles)
    return mask_circles

def detect_and_score(img, gt_mask):
    mask = detect_apples(img)
    return get_score(mask, gt_mask)
