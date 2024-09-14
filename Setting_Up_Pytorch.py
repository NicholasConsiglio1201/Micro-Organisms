import torch
# Importing the ability use the Yolo Model for Object Detection
from ultralytics.models import YOLO
import cv2

print("Hello")

source = "C:/Users/ncons/AppData/Local/Programs/Python/Python39/Scripts"
 
# Loading in the Model we will be using for Object Detection
model = YOLO('yolov8n.pt')
 
video_path = "./test.mp4"
cap = cv2.VideoCapture(video_path)
 
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # Detect Objects
    
    # Track Objects
    results = model.track(frame, persist = True)
 
    # Plot Results
    frame_ = results[0].plot()
 
    # Visualize Results
    cv2.imshow("Frame", frame_)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

