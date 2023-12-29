from ultralytics import YOLO
import cv2

# Load an official or custom model
model = YOLO('yolov8n.pt')  # Load an official Detect model

# Perform tracking with the model
results = model.track(source="cu6896.mp4", show=True, tracker="bytetrack.yaml")  # Tracking with ByteTrack tracker
for r in results:
        boxes = r.boxes  # Boxes object for bbox outputs
        masks = r.masks  # Masks object for segment masks outputs
        probs = r.probs  # Class probabilities for classification outputs

cv2.waitKey(0)
cv2.destroyAllWindows()