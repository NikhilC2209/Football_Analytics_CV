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
from ultralytics import YOLO
from crop_filtering import *

# Load the YOLOv8 model
model = YOLO('runs/detect/train2/weights/best.pt')

# Open the video file
#video_path = "clips/40cd38_1.mp4"
video_path = "sample/MAN_Derby_sample.mkv"
cap = cv2.VideoCapture(video_path)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True, tracker="botsort.yaml")

        for r in results:
            all_labels = r.boxes.cls.tolist()
            all_coords_list = r.boxes.xyxy.tolist()

        print(all_labels)

        player_coords_list = [all_coords_list[i] for i in range(len(all_coords_list)) if all_labels[i]==1]          # Label 1 refers to Player Label

        x1, y1, x2, y2 = player_coords_list[0]

        # print(x1, y1, x2, y2)

        # SKY-BLUE MASK (MAN CITY)

        team1_hsv = (np.array((80, 0, 120)), np.array((110, 255, 255))) 

        # RED MASK (MAN UTD)

        team2_hsv = (np.array((169, 0, 0)), np.array((179, 255, 255)))

        # GREEN MASK (MAN CITY & MAN UTD GK)

        gk1_hsv = (np.array((35, 0, 151)), np.array((80, 255, 255)))    
        gk2_hsv = (np.array((35, 0, 151)), np.array((80, 255, 255)))    

        # CROPPED IMAGE
        img_box = frame[int(y1): int(y2), int(x1): int(x2)]

        h, w, _ = img_box.shape

        crop_img = img_box[int(0.3*h): int(0.5*h), int(0.4*w): int(0.6*w)]

        cv2.imshow("Cropped", crop_img)

        all_masks = get_all_masks(crop_img, team1_hsv, team2_hsv, gk1_hsv, gk2_hsv)

        count1, count2, count3, count4 = get_pixel_count(crop_img, all_masks)

        labels = {count1: "MAN_CITY", count2: "MAN_UTD", count3: "MAN_CITY_GK", count4: "MAN_UTD_GK"}
        
        #labels = get_labels("MAN_CITY", "MAN_UTD", "MAN_CITY_GK", "MAN_UTD_GK")
        color_labels = {"MAN_CITY": (135, 206, 235), "MAN_UTD": (0,0,255), "MAN_CITY_GK": (0,255,0), "MAN_UTD_GK": (0,255,0)}    # BGR FORMAT, NOT HSV
        
        crop_label = labels.get(max(labels))

        print(crop_label)
        print(color_labels[crop_label])

        img = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color_labels[crop_label], 1)
        
        #cv2.imshow("Cropped", crop_img)
        cv2.imshow("YOLOv8 Tracking", img)

        # cv2.imshow("Cropped Image", crop_img)
        # cv2.imwrite("results/gk.jpg", crop_img)

        # Visualize the results on the frame
        #annotated_frame = results[0].plot(boxes=True, conf=False, labels=False, line_width=2)

        # Display the annotated frame
        #cv2.imshow("YOLOv8 Tracking", annotated_frame)
        #cv2.imwrite("results/frame.jpg", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()


##################

# Next: Use a for loop to get bounding boxes around each player in every frame

##################