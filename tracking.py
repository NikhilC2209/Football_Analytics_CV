# import cv2
# from ultralytics import YOLO

# # PRE-TRAINED MODEL
# model = YOLO('runs/detect/train2/weights/best.pt')  

# INFERENCE
#results = model.track(source="clips/08fd33_0.mp4", show=True, save_crop=True, tracker="botsort.yaml")
#results = model.track(source="sample/MAN_Derby_sample.mkv", show=True, save_crop=True, tracker="botsort.yaml")

# # Show the results
# for r in results:
#     im_array = r.plot()  # plot a BGR numpy array of predictions
#     im.show()  # show image
#     im.save('results.jpg')  # save image

import cv2
import pandas as pd
from ultralytics import YOLO
from crop_filtering import *

# Load the YOLOv8 model
model = YOLO('runs/detect/train4/weights/best.pt')

# Open the video file
#video_path = "clips/40cd38_1.mp4"
video_path = "sample/Liverpool_vs_Arsenal.mkv"
df = pd.read_csv("hsv_codes.csv")
cap = cv2.VideoCapture(video_path)

def get_match_name(filename):
    return filename.split("/")[1].split(".")[0]

def get_teams(filename):
    match_name = filename.split("/")[1].split(".")[0]
    team1, team2 = match_name.split("_vs_")

    print(team1, team2)

    return team1, team2

def load_masks(df, team1, team2):

    team1_codes = df['Home'][df['Team']==team1].item()
    team2_codes = df['Away'][df['Team']==team2].item()

    team1_gk = df['GK1'][df['Team']==team1].item()
    team2_gk = df['GK2'][df['Team']==team2].item()

    # team1_lower_mask, team1_upper_mask = team1.split(";")
    # team2_lower_mask, team2_upper_mask = team2.split(";")

    #return team1_lower_mask, team1_upper_mask, team2_lower_mask, team2_upper_mask, team1_gk, team2_gk
    return eval(team1_codes), eval(team2_codes), eval(team1_gk), eval(team2_gk)             # eval() to convert from string to tuple

def get_color_labels():

    team1 = (0,0,255)
    team2 = (0,255,255)

    gk1 = (0,0,255)
    gk2 = (0,255,255)

    ball = (128,0,128)
    referee = (0,0,0)

    return team1, team2, gk1, gk2

def multi_key_dict_get(d, k):
    for keys, v in d.items():
        if k in keys:
            return v
    return None

### LIV vs ARS

team1, team2 = get_teams(video_path)
# team1_lower_mask, team1_upper_mask, team2_lower_mask, team2_upper_mask, team1_gk, team2_gk = load_masks(df, team1, team2)

team1_hsv, team2_hsv, gk1_hsv, gk2_hsv = load_masks(df, team1, team2)

# team1_hsv = (np.array(team1_lower_mask), np.array(team1_upper_mask))
# team2_hsv = (np.array(team2_lower_mask), np.array(team2_upper_mask))

# # SKY-BLUE MASK (MAN CITY)

# team1_hsv = (np.array((80, 0, 120)), np.array((110, 255, 255))) 

# # RED MASK (MAN UTD)

# team2_hsv = (np.array((169, 0, 0)), np.array((179, 255, 255)))

# # GREEN MASK (MAN CITY & MAN UTD GK)

# gk1_hsv = (np.array((35, 0, 151)), np.array((80, 255, 255)))    
# gk2_hsv = (np.array((35, 0, 151)), np.array((80, 255, 255)))   
        
# #labels = get_labels("MAN_CITY", "MAN_UTD", "MAN_CITY_GK", "MAN_UTD_GK")
# color_labels = {"MAN_CITY": (235, 206, 135), "MAN_UTD": (0,0,255), "MAN_CITY_GK": (0,255,0), "MAN_UTD_GK": (0,255,0), "BALL": (128,0,128), "REFEREE": (0,0,0)}    # BGR FORMAT, NOT HSV
color_labels = {("Liverpool", "Liverpool_GK") : (0,0,255), ("Arsenal", "Arsenal_GK") : (0,255,255), "Ball": (128,0,128), "Referee": (0,0,0)}    # BGR FORMAT, NOT HSV

output_clip = cv2.VideoWriter("results/" + get_match_name(video_path) + ".avi", cv2.VideoWriter_fourcc(*'MJPG'), 60, (1920, 1080)) 

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    #print(frame.shape)
    # fps = int(cap.get(cv2.CAP_PROP_FPS))
    # print("fps: ", fps)

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True, tracker="botsort.yaml")

        for r in results:
            all_labels = r.boxes.cls.tolist()
            all_coords_list = r.boxes.xyxy.tolist()
            all_conf_scores = r.boxes.conf.tolist() 

        #print(all_conf_scores)

        player_coords_list = [(all_coords_list[i], str(round(all_conf_scores[i],2))) for i in range(len(all_coords_list)) if all_labels[i]==1]          # Label 1 refers to Player Label
        referee_coords_list = [(all_coords_list[i], str(round(all_conf_scores[i],2))) for i in range(len(all_coords_list)) if all_labels[i]==2]         # Label 2 refers to Referee Label
        ball_coords_list = [(all_coords_list[i], str(round(all_conf_scores[i],2))) for i in range(len(all_coords_list)) if all_labels[i]==0]            # Label 0 refers to Ball Label

        for (ref, score) in referee_coords_list:
            x1, y1, x2, y2 = ref
            img = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color_labels["Referee"], 2)

            (text_w, text_h), _ = cv2.getTextSize("Referee: " + score, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)

            # PRINT TEXT ON BOUNDING BOXES
            img = cv2.rectangle(img, (int(x1), int(y1 - 20)), (int(x1 + text_w), int(y1)), color_labels["Referee"], -1)
            cv2.putText(img, "Referee: " + score, (int(x1), int(y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        for (ball, score) in ball_coords_list:
            x1, y1, x2, y2 = ball
            img = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color_labels["Ball"], 2)

            (text_w, text_h), _ = cv2.getTextSize("Ball: " + score, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)

            # PRINT TEXT ON BOUNDING BOXES
            img = cv2.rectangle(img, (int(x1), int(y1 - 20)), (int(x1 + text_w), int(y1)), color_labels["Ball"], -1)
            cv2.putText(img, "Ball: " + score, (int(x1), int(y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        
        for (player, score) in player_coords_list:
            x1, y1, x2, y2 = player

            # CROPPED IMAGE
            img_box = frame[int(y1): int(y2), int(x1): int(x2)]

            h, w, _ = img_box.shape
            crop_img = img_box[int(0.3*h): int(0.5*h), int(0.4*w): int(0.6*w)]
            cv2.imshow("Cropped", crop_img)

            all_masks = get_all_masks(crop_img, team1_hsv, team2_hsv, gk1_hsv, gk2_hsv)
            count1, count2, count3, count4 = get_pixel_count(crop_img, all_masks)

            # labels = {count1: "MAN_CITY", count2: "MAN_UTD", count3: "MAN_CITY_GK", count4: "MAN_UTD_GK"}

            labels = {count1: team1, count2: team2, count3: team1+"_GK", count4: team2+"_GK"}

            crop_label = labels.get(max(labels))
            final_label = multi_key_dict_get(color_labels, crop_label)
            #print(final_label)

            # PLOT BOUNDING BOXES
            img = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), final_label, 2)

            (text_w, text_h), _ = cv2.getTextSize(crop_label + ": " + score, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)

            # PRINT TEXT ON BOUNDING BOXES
            img = cv2.rectangle(img, (int(x1), int(y1 - 20)), (int(x1 + text_w), int(y1)), final_label, -1)
            cv2.putText(img, crop_label + ": " + score, (int(x1), int(y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

        output_clip.write(img)

        #cv2.imshow("Cropped", crop_img)
        cv2.imshow("YOLOv8 Tracking", img)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
output_clip.release()
cv2.destroyAllWindows()


##################

# Next: Use a for loop to get bounding boxes around each player in every frame

##################