import cv2
from ultralytics import YOLO
import time

# Load the YOLOv8 model
# model = YOLO('yolov8n.pt')          ### Pre-trained weights

model = YOLO('runs/detect/train2/weights/best.pt')          ### Custom weights

# Open the video file
video_path = "sample/ARS_VS_LENS.mp4"
cap = cv2.VideoCapture(video_path)

fps_start_time = 0
fps = 0

# INFERENCE

results = model.track(source="sample/MAN_Derby.mkv", persist=True, show=True, save=True, save_crop=True, tracker="botsort.yaml", line_width=2)

# Loop through the video frames
# while cap.isOpened():
#     # Read a frame from the video
#     success, frame = cap.read()

#     # fps_end_time = time.time()
#     # time_diff = fps_end_time - fps_start_time
#     # fps = int(1/(time_diff))
#     # fps_start_time = fps_end_time

#     # font = cv2.FONT_HERSHEY_SIMPLEX
#     # cv2.putText(frame, str(fps), (50,50), font, 1, (0,0,255), 2)

#     if success:
#         # Run YOLOv8 tracking on the frame, persisting tracks between frames
#         #results = model.track(frame, persist=True, show=True, save=True, save_crop=True, tracker="botsort.yaml", conf=False, show_labels=False, line_width=2)
#         results = model.track(frame, persist=True, show=True, save=True, tracker="bytetrack.yaml")

#         # Visualize the results on the frame
#         #annotated_frame = results[0].plot()

#         # Display the annotated frame
#         #cv2.imshow("YOLOv8 Tracking", annotated_frame)

#         # Break the loop if 'q' is pressed
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break
#     else:
#         # Break the loop if the end of the video is reached
#         break

# # Release the video capture object and close the display window
# cap.release()
# cv2.destroyAllWindows()
