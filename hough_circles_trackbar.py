import cv2
import numpy as np

def nothing(x): pass
cv2.namedWindow('image')

img_path = 'apple.jpg'

cimg = cv2.imread(img_path)
gimg = cv2.cvtColor(cimg, cv2.COLOR_BGR2GRAY)

# Create trackbars for color change
# Hue is from 0-179 for Opencv
cv2.createTrackbar('HMin', 'image', 0, 179, nothing)
cv2.createTrackbar('HMax', 'image', 0, 179, nothing)

cv2.createTrackbar('SMin', 'image', 0, 255, nothing)
cv2.createTrackbar('SMax', 'image', 0, 255, nothing)

cv2.createTrackbar('VMin', 'image', 0, 255, nothing)
cv2.createTrackbar('VMax', 'image', 0, 255, nothing)

cv2.createTrackbar('Param 1','image', 1, 500, nothing)
cv2.createTrackbar('Param 2','image', 1, 100, nothing)
cv2.createTrackbar('Min dist','image', 1, 1000, nothing)
cv2.createTrackbar('Min rad','image', 1, 100, nothing)
cv2.createTrackbar('Max rad','image', 1, 1000, nothing)

cv2.setTrackbarPos('HMin', 'image', 140)
cv2.setTrackbarPos('HMax', 'image', 5)
cv2.setTrackbarPos('SMax', 'image', 255)
cv2.setTrackbarPos('VMax', 'image', 255)

cv2.setTrackbarPos('Param 1','image', 118)
cv2.setTrackbarPos('Param 2','image', 33)
cv2.setTrackbarPos('Min dist','image', 180)
cv2.setTrackbarPos('Min rad','image', 7)
cv2.setTrackbarPos('Max rad','image', 133)

hMin = sMin = vMin = hMax = sMax = vMax = 0
phMin = psMin = pvMin = phMax = psMax = pvMax = 0

while True:
    hMin = cv2.getTrackbarPos('HMin', 'image')
    hMax = cv2.getTrackbarPos('HMax', 'image')
    sMin = cv2.getTrackbarPos('SMin', 'image')
    sMax = cv2.getTrackbarPos('SMax', 'image')
    vMin = cv2.getTrackbarPos('VMin', 'image')
    vMax = cv2.getTrackbarPos('VMax', 'image')

    lower = np.array([hMin, sMin, vMin])
    upper = np.array([hMax, sMax, vMax])

    #cimg[..., ::-1]
    hsv = cv2.cvtColor(cimg, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    mask = (h<=hMax)|(h>=hMin) if hMin>hMax else (hMin<=h)&(h<=hMax)
    mask &= (sMin<=s)&(s<=sMax) & (vMin<=v)&(s<=vMax)
    mask = mask.astype(np.uint8)
    mask[mask==1] = 255

    # mask = cv2.inRange(hsv, lower, upper)
    result = cv2.bitwise_and(cimg, cimg, mask=mask)

    ## final mask and masked
    # mask = cv2.bitwise_or(mask1, mask2)

    result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    result[result==0] = 255

    param1 = cv2.getTrackbarPos('Param 1','image')
    param2 = cv2.getTrackbarPos('Param 2','image')
    min_dist = cv2.getTrackbarPos('Min dist','image')
    min_radius = cv2.getTrackbarPos('Min rad','image')
    max_radius = cv2.getTrackbarPos('Max rad','image')
    circles = cv2.HoughCircles(
                        result,
                        cv2.HOUGH_GRADIENT,
                        2,   # accumulator resolution (size of the image / 2)
                        min_dist,  # minimum distance between two circles
                        120,
                        param1, # Canny high threshold
                        param2, # minimum number of votes
                        minRadius=min_radius,
                        maxRadius=max_radius
    )

    circles = np.uint16(np.around(circles))
    cimg2 = cv2.bitwise_and(cimg, cimg, mask=mask)
    cimg2[result==0] = 255
    for i in circles[0,:]:
        cv2.circle(cimg2,(i[0],i[1]),i[2],(0,255,0),2)
        cv2.circle(cimg2,(i[0],i[1]),2,(0,0,255),3)

    cv2.imshow('image', cv2.resize(cimg2,[1200,600]))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
