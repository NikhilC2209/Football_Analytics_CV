import cv2
import numpy as np

# Load Cropped Image
img = cv2.imread("results/gk.jpg")
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# SKY-BLUE MASK (MAN CITY)

lower_sky_blue = np.array((80, 0, 120))
upper_sky_blue = np.array((110, 255, 255))

# RED MASK (MAN UTD)

lower_red = np.array((169, 0, 0))
upper_red = np.array((179, 255, 255))

# GREEN MASK (MAN CITY & MAN UTD GK)

lower_green = np.array((35, 0, 151))
upper_green = np.array((80, 255, 255))

sky_blue_mask = cv2.inRange(hsv_img, lower_sky_blue, upper_sky_blue)
red_mask = cv2.inRange(hsv_img, lower_red, upper_red)
green_mask = cv2.inRange(hsv_img, lower_green, upper_green)

out_red = cv2.bitwise_and(img, img, mask=red_mask)
out_sky_blue = cv2.bitwise_and(img, img, mask=sky_blue_mask)
out_green = cv2.bitwise_and(img, img, mask=green_mask)

count1 = np.sum(out_red != 0)
count2 = np.sum(out_sky_blue != 0)
count3 = np.sum(out_green != 0)

labels = {count1: "MAN_UTD", count2: "MAN_CITY", count3: "GOALKEEPER"}

print(labels.get(max(labels)))

# Calculate no. of non-black pixels from all masks and label the crop image with that team's mask
# count = np.sum(out != 0)
# print(count)

cv2.imshow("Output", img)

cv2.waitKey(0)
cv2.destroyAllWindows() 

