from collections import defaultdict
import json
import cv2
import numpy as np

from ultralytics import YOLO

def get_keypoints_from_orb(image):
    # Inicializamos el detector de keypoints
    orb = cv2.ORB_create()

    # Detectamos los keypoints en la imagen
    keypoints, descriptors = orb.detectAndCompute(image, None)

    return keypoints, descriptors

# Cargamos la imagen
image = cv2.imread("img_1.PNG")

# Obtenemos los keypoints y las descriptors
keypoints, descriptors = get_keypoints_from_orb(image)

# Imprimimos los keypoints
print(f'keypoints: {len(keypoints)}')
print(f'descriptors: {descriptors[0]}')

keypoints = np.array(keypoints)
#keypoints = keypoints.reshape((keypoints.shape[0], 2))

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Open the video file
video_path = "cu6896.mp4"
cap = cv2.VideoCapture(video_path)

# Store the track history
track_history = defaultdict(lambda: [])

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)

        # Get the boxes and track IDs
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Plot the tracks
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(x), float(y)))  # x, y center point
            if len(track) > 30:  # retain 90 tracks for 90 frames
                track.pop(0)

            # Draw the tracking lines
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()

print(keypoints)

# Nombre del archivo JSON
nombre_archivo = 'track_history.json'

# Escribir el diccionario en el archivo JSON
with open(nombre_archivo, 'w') as archivo_json:
    json.dump(track_history, archivo_json)