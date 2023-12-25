import cv2
import pandas as pd

sample_img = cv2.imread("crops/reference.jpg")

team1_player = cv2.imread("crops/Man_City_player.jpg")
team2_player = cv2.imread("crops/Man_UTD_player.jpg")

df = pd.read_csv("hsv_codes.csv")

team1 = df['Home'][df['Team']=="Arsenal"].item()
team2 = df['Away'][df['Team']=="Liverpool"].item()

team1_lower_mask, team1_upper_mask = team1.split(";")
team2_lower_mask, team2_upper_mask = team2.split(";")

print(team1_lower_mask, team1_upper_mask)
print(team2_lower_mask, team2_upper_mask)

# # def sim_score(img, ref):
# # 	random_sample =

# def return_crop(img):
# 	h, w, _ = img.shape
# 	crop_img = img[int(0.3*h): int(0.5*h), int(0.4*w): int(0.6*w)] 

# 	return crop_img

# cv2.imshow("Cropped", return_crop(sample_img))
# cv2.imshow("Original", sample_img)


# cv2.waitKey(0)
# cv2.destroyAllWindows() 