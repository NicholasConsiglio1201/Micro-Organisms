from ultralytics import YOLO
import cv2

# Load Model
#model = YOLO("yolov8n.yaml") # Building a Yolov8 model from scratch
model = YOLO(r"C:\Users\ncons\OneDrive\Documents\Object Detection\runs\detect\train11\weights\best.pt")

# Use the Model
#results = model.train(data = "config.yaml", epochs = 25) # Training the Model

video_path = "./minish_cap_hat.mp4"
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
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()