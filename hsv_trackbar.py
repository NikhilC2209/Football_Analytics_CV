import cv2
import numpy as np

#frame = cv2.imread("sample/img1.jpg") 

def nothing(x):
    pass

cv2.namedWindow('marking')

lowH = 0
highH = 179
lowS = 0
highS = 255
lowV = 0
highV = 255

# create trackbars for color change
cv2.createTrackbar('lowH','marking', lowH, 179,nothing)
cv2.createTrackbar('highH','marking', highH, 179,nothing)

cv2.createTrackbar('lowS','marking', lowS, 255,nothing)
cv2.createTrackbar('highS','marking', highS, 255,nothing)

cv2.createTrackbar('lowV','marking', lowV, 255,nothing)
cv2.createTrackbar('highV','marking', highV, 255,nothing)

while True:

    frame = cv2.imread("results/gk.jpg") 
    #frame = cv2.resize(frame, (768, 540))  

    ilowH = cv2.getTrackbarPos('lowH', 'marking')
    ihighH = cv2.getTrackbarPos('highH', 'marking')
    ilowS = cv2.getTrackbarPos('lowS', 'marking')
    ihighS = cv2.getTrackbarPos('highS', 'marking')
    ilowV = cv2.getTrackbarPos('lowV', 'marking')
    ihighV = cv2.getTrackbarPos('highV', 'marking')

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_hsv = np.array([ilowH, ilowS, ilowV])
    higher_hsv = np.array([ihighH, ihighS, ihighV])
    mask = cv2.inRange(hsv, lower_hsv, higher_hsv)

    frame = cv2.bitwise_and(frame, frame, mask=mask)

    # show thresholded image
    cv2.imshow('image', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()