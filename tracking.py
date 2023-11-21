# import cv2
# from ultralytics import YOLO

# # PRE-TRAINED MODEL
# model = YOLO('runs/detect/train2/weights/best.pt')  

# # INFERENCE
# results = model.track(source="clips/08fd33_0.mp4", show=True, tracker="botsort.yaml")

# # Show the results
# for r in results:
#     im_array = r.plot()  # plot a BGR numpy array of predictions
#     im.show()  # show image
#     im.save('results.jpg')  # save image

import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('runs/detect/train2/weights/best.pt')

# Open the video file
#video_path = "clips/40cd38_1.mp4"
video_path = "sample/MAN_Derby.mkv"
cap = cv2.VideoCapture(video_path)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True, tracker="botsort.yaml")

        for r in results:
        	annotation_list = r.boxes.xyxy

        x1, y1, x2, y2 = annotation_list[0].tolist()

        print(x1, y1, x2, y2)

        crop_img = frame[int(y1): int(y2), int(x1): int(x2)]

        cv2.imshow("Cropped Image", crop_img)
        cv2.imwrite("results/frame.jpg", crop_img)

        # Visualize the results on the frame
        annotated_frame = results[0].plot(boxes=True, conf=False, labels=False, line_width=2)

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

# Next: Use the cropped image in the hsv_trackbar

##################