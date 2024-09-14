# Imports the Ultralytics and YOLO model that we will be using for our object detection
from ultralytics import YOLO
# Imports the ability to use the OpenCV library -- CV stands for Computer Vision
import cv2

# Here we are defining the path that we want our YOLO v8 model to use. This path contains special weights that allow us to identify Link
model_path = r"C:\Users\ncons\OneDrive\Documents\Object Detection\runs\detect\train14\weights\best.pt"
# This here is the model we will be using for our object detection
model = YOLO(model_path)

# Path of the Video that we want to use object detection on
video_path = "./Acartia_Red_Control_Test_2_05232024.mp4"

# Capturing the frames of the video and opens the Video file for reading.
cap = cv2.VideoCapture(video_path)

# Doing some error handling here to see if the video file is being opened correctly
if cap.isOpened():
    print("The video file is open and ready to be tracked!")
else:
    print("The video is not opening properly.")

# Getting the Width, Height, and Frames per second of the video
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

print("----------------------------------------------------")
print("This is the value of the frame width:", frame_width)
print("This is the value of the frame height:", frame_height)
print("This is the value of the original frames per second of the video:", fps)

# Specifying a path for the output file
output_path = r"C:\Users\ncons\OneDrive\Documents\Object Detection\tracked_output2.mp4"
# Encodes the output for the video
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for .mp4 file
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
 
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # Detect Objects
    
    # Track Objects
    results = model(frame)
    ###results = model.track(frame, persist = True)
 
    # Plot Results
    frame_ = results[0].plot()

    # Write the frame to the output video file
    out.write(frame_)
 
    # Visualize Results
    cv2.imshow("Frame", frame_)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()