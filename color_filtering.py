import cv2
import numpy as np

# Open Image

sample_img = cv2.imread("sample/img1.jpg") 
sample_img = cv2.resize(sample_img, (768, 540))  

# In-Range Filtering

hsv_img = cv2.cvtColor(sample_img, cv2.COLOR_BGR2HSV)

mask1 = cv2.inRange(hsv_img, (0, 200, 20), (10, 255, 255))
mask2 = cv2.inRange(hsv_img, (160, 200, 20), (179, 255, 255))
#inv_mask = cv2.bitwise_not(mask)

mask = mask1 + mask2

out = cv2.bitwise_and(sample_img, sample_img, mask=mask)
#inv_out = cv2.bitwise_and(sample_img, sample_img, inv_mask)

cv2.imshow("Image", sample_img)
cv2.imshow("Mask", mask)
cv2.imshow("Out", out)

cv2.waitKey(0)
cv2.destroyAllWindows() 

### WEBCAM STREAM

# cap = cv2.VideoCapture(0)

# while True:
# 	_, frame = cap.read()
# 	hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


# 	mask1 = cv2.inRange(hsv_img, (0, 100, 20), (10, 255, 255))
# 	mask2 = cv2.inRange(hsv_img, (160, 100, 20), (179, 255, 255))

# 	# lower = np.array([0,0,0])
# 	# upper = np.array([60,255,255])

# 	mask = mask1 + mask2
# 	out = cv2.bitwise_and(frame, frame, mask=mask)

# 	cv2.imshow("Image", frame)
# 	cv2.imshow("Mask", mask)
# 	cv2.imshow("Out", out)

# 	if cv2.waitKey(1) & 0xFF == ord("q"):
# 		break



#cv2.waitKey(0)
# cv2.destroyAllWindows() 
# cap.release()