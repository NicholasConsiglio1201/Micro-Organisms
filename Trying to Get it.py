from collections import defaultdict
from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd

# Load Model
model = YOLO(r"C:\Users\ncons\OneDrive\Documents\Object Detection\runs\detect\train11\weights\best.pt")

video_path = "./minish_cap_hat.mp4"
cap = cv2.VideoCapture(video_path)

# Store the track history
track_history = defaultdict(lambda: [])
track_missing = defaultdict(int)
max_track_length = 300
max_missing_frames = 10 

# Initialize DataFrame to store tracking points
columns = ['track_id', 'frame', 'x', 'y']
df = pd.DataFrame(columns=columns)

frame_number = 0

while cap.isOpened():
    ret, frame = cap.read()
    frame_number += 1
    
    if not ret:
        print("Failed to read frame or reached end of video!")
        break

    # Run Yolov8 tracking on the frame, persisting tracks between frames
    results = model.track(frame, persist=True)

    result = results[0] if results else None

    # Check if there are any detections
    if result and result.boxes is not None and result.boxes.id is not None:
        boxes = result.boxes.xywh.cpu().numpy() if result.boxes.xywh is not None else []
        track_ids = result.boxes.id.int().cpu().tolist() if result.boxes.id is not None else []
        classes = result.boxes.cls.int().cpu().tolist() if result.boxes.cls is not None else []

        # Filter for "Link" class detections (assuming class index 0 for "Link")
        link_boxes = [box for box, cls in zip(boxes, classes) if cls == 0]
        link_ids = [track_id for track_id, cls in zip(track_ids, classes) if cls == 0]

        if link_boxes:
            # Reset missing counter for detected objects
            for track_id in link_ids:
                track_missing[track_id] = 0

            # Visualize the results on the frame
            annotated_frame = result.plot()

            # Update the track history and DataFrame only if "Link" detections are found
            new_rows = []
            for box, track_id in zip(link_boxes, link_ids):
                x, y, w, h = box
                track = track_history[track_id]
                track.append((float(x), float(y))) # x, y center point

                # Create a new row for the DataFrame
                new_row = {'track_id': track_id, 'frame': frame_number, 'x': float(x), 'y': float(y)}
                new_rows.append(new_row)

                # Limit track history length
                if len(track) > max_track_length:
                    track.pop(0)
            
            # Concatenate the new rows to the DataFrame
            df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        else:
            # No "Link" detections in this frame
            annotated_frame = frame
    else:
        # If no objects detected, increment the missing frame counter
        for track_id in list(track_history.keys()):
            track_missing[track_id] += 1
            if track_missing[track_id] > max_missing_frames:
                del track_history[track_id]
                del track_missing[track_id]

        # Use the original frame if no detections to maintain the video output
        annotated_frame = frame

    # Draw the tracking lines for all tracks
    for track_id, track in track_history.items():
        if len(track) > 1:
            points = np.array(track, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [points], isClosed=False, color=(0, 0, 0), thickness=10)

    # Display the annotated frame
    cv2.imshow("Yolov8 Tracking", annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(30) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

# Print DataFrame for debugging purposes
print(df)

# Optionally, save DataFrame to a CSV file
df.to_csv('tracking_points.csv', index=False)