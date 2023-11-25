import cv2
import numpy as np

# # Load Cropped Image
# img = cv2.imread("results/gk.jpg")
# hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# # SKY-BLUE MASK (MAN CITY)

# lower_sky_blue = np.array((80, 0, 120))
# upper_sky_blue = np.array((110, 255, 255))

# # RED MASK (MAN UTD)

# lower_red = np.array((169, 0, 0))
# upper_red = np.array((179, 255, 255))

# # GREEN MASK (MAN CITY & MAN UTD GK)

# lower_green = np.array((35, 0, 151))
# upper_green = np.array((80, 255, 255))

# sky_blue_mask = cv2.inRange(hsv_img, lower_sky_blue, upper_sky_blue)
# red_mask = cv2.inRange(hsv_img, lower_red, upper_red)
# green_mask = cv2.inRange(hsv_img, lower_green, upper_green)

# out_red = cv2.bitwise_and(img, img, mask=red_mask)
# out_sky_blue = cv2.bitwise_and(img, img, mask=sky_blue_mask)
# out_green = cv2.bitwise_and(img, img, mask=green_mask)

# count1 = np.sum(out_red != 0)
# count2 = np.sum(out_sky_blue != 0)
# count3 = np.sum(out_green != 0)

# labels = {count1: "MAN_UTD", count2: "MAN_CITY", count3: "GOALKEEPER"}

# print(labels.get(max(labels)))

# # Calculate no. of non-black pixels from all masks and label the crop image with that team's mask
# # count = np.sum(out != 0)
# # print(count)

# cv2.imshow("Output", img)

# cv2.waitKey(0)
# cv2.destroyAllWindows() 

def get_all_masks(img, team1_hsv, team2_hsv, gk1_hsv, gk2_hsv):

	lower_team1, upper_team1 = team1_hsv
	lower_team2, upper_team2 = team2_hsv
	lower_gk1, upper_gk1 = gk1_hsv
	lower_gk2, upper_gk2 = gk2_hsv

	hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

	team1_mask = cv2.inRange(hsv_img, lower_team1, upper_team1)
	team2_mask = cv2.inRange(hsv_img, lower_team2, upper_team2)
	gk1_mask = cv2.inRange(hsv_img, lower_gk1, upper_gk1)
	gk2_mask = cv2.inRange(hsv_img, lower_gk2, upper_gk2)

	return team1_mask, team2_mask, gk1_mask, gk2_mask

def get_labels(team1_name, team2_name, gk1, gk2):

	labels = {count1: team1_name, count2: team2_name, count3: gk1, count4: gk2}

	return labels

def get_pixel_count(img, all_masks):

	team1_mask, team2_mask, gk1_mask, gk2_mask = all_masks

	out_team1 = cv2.bitwise_and(img, img, mask=team1_mask)
	out_team2 = cv2.bitwise_and(img, img, mask=team2_mask)
	out_gk1 = cv2.bitwise_and(img, img, mask=gk1_mask)
	out_gk2 = cv2.bitwise_and(img, img, mask=gk2_mask)

	count1 = np.sum(out_team1 != 0)
	count2 = np.sum(out_team2 != 0)
	count3 = np.sum(out_gk1 != 0)
	count4 = np.sum(out_gk2 != 0)

	return (count1, count2, count3, count4)