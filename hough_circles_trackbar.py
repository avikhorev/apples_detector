import cv2
import numpy as np

PAR={
 'equalize_hist': False,
 'binarize':True
}

def detect_apples(cimg):
    blur = PAR['blur'];
    if blur%2==0: blur+=1
    cimg = cv2.GaussianBlur(cimg, (blur,blur), 0)
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
    mask = (h<=hMax)|(h>=hMin) if hMin>hMax else (hMin<=h)&(h<=hMax)
    mask &= (sMin<=s)&(s<=sMax) & (vMin<=v)&(s<=vMax)
    mask = mask.astype(np.uint8)
    cimg = cv2.bitwise_and(cimg, cimg, mask=mask)

    gray = cv2.cvtColor(cimg, cv2.COLOR_BGR2GRAY)
    # result[result==0] = 255

    if PAR['binarize']:
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)[1]

    dp = PAR['dp']
    param1 = PAR['param1']
    param2 = PAR['param2']
    min_dist = PAR['min_dist']
    min_radius = PAR['min_radius']
    max_radius = PAR['max_radius']

    circles = cv2.HoughCircles(
                        gray,
                        cv2.HOUGH_GRADIENT,
                        dp=dp,   # accumulator resolution (image_size/dp)
                        minDist=min_dist,  # minimum distance between two circles
                        param1=param1, # canny high threshold
                        param2=param2, # minimum number of votes
                        minRadius=min_radius,
                        maxRadius=max_radius
    )
    if circles is None:
        return cimg, []
    circles = np.uint16(circles[0,:])
    # gray_lo, gray_hi = np.array([70, 70, 70]), np.array([200, 200, 200])
    # gray_mask = cv2.bitwise_and(gray, gray, mask=cv2.inRange(cimg, gray_lo, gray_hi))
    res = []
    for circle in circles:
        if len(circle.shape)==0:
            continue
        x, y, r = circle
        res.append(circle)
        # if gray_mask[y-1, x-1] == 0:
    return cimg, res


def nothing(x): pass
cv2.namedWindow('image')

img_path = 'apple.jpg'
# img_path = 'photo_2021-12-17_20-42-08.jpg'

cimg_orig = cv2.imread(img_path)
gimg = cv2.cvtColor(cimg_orig, cv2.COLOR_BGR2GRAY)

# Hue is from 0-179 for Opencv
cv2.createTrackbar('HMin', 'image', 0, 179, nothing)
cv2.createTrackbar('HMax', 'image', 0, 179, nothing)

cv2.createTrackbar('SMin', 'image', 0, 255, nothing)
cv2.createTrackbar('SMax', 'image', 0, 255, nothing)

cv2.createTrackbar('VMin', 'image', 0, 255, nothing)
cv2.createTrackbar('VMax', 'image', 0, 255, nothing)

cv2.createTrackbar('Blur','image', 1, 30, nothing)
cv2.createTrackbar('Param 1','image', 1, 500, nothing)
cv2.createTrackbar('Param 2','image', 1, 100, nothing)
cv2.createTrackbar('Min dist','image', 1, 1000, nothing)
cv2.createTrackbar('Min rad','image', 1, 100, nothing)
cv2.createTrackbar('Max rad','image', 1, 100, nothing)

cv2.setTrackbarPos('HMin', 'image', 140)
cv2.setTrackbarPos('HMax', 'image', 5)
cv2.setTrackbarPos('SMax', 'image', 255)
cv2.setTrackbarPos('VMax', 'image', 255)

cv2.setTrackbarPos('Blur','image', 5)

cv2.setTrackbarPos('Param 1','image', 101)
cv2.setTrackbarPos('Param 2','image', 8)
cv2.setTrackbarPos('Min dist','image', 42)
cv2.setTrackbarPos('Min rad','image', 9)
cv2.setTrackbarPos('Max rad','image', 16)

while True:

    cimg = cimg_orig

    PAR['hMin'] = cv2.getTrackbarPos('HMin', 'image')
    PAR['hMax'] = cv2.getTrackbarPos('HMax', 'image')
    PAR['sMin'] = cv2.getTrackbarPos('SMin', 'image')
    PAR['sMax'] = cv2.getTrackbarPos('SMax', 'image')
    PAR['vMin'] = cv2.getTrackbarPos('VMin', 'image')
    PAR['vMax'] = cv2.getTrackbarPos('VMax', 'image')

    PAR['blur'] = cv2.getTrackbarPos('Blur', 'image')

    PAR['dp'] = 2
    PAR['param1'] = cv2.getTrackbarPos('Param 1','image')
    PAR['param2'] = cv2.getTrackbarPos('Param 2','image')
    PAR['min_dist'] = cv2.getTrackbarPos('Min dist','image')
    PAR['min_radius'] = cv2.getTrackbarPos('Min rad','image')
    PAR['max_radius'] = cv2.getTrackbarPos('Max rad','image')

    cimg, circles = detect_apples(cimg)
    for x,y,r in circles:
        cv2.circle(cimg, (x,y), r, (255,255,0), thickness=2)
        cv2.circle(cimg, (x,y), 2, (0,255,255), thickness=-1)

    cv2.imshow('image', cv2.resize(cimg,[1200,600]))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
