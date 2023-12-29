import cv2
import numpy as np
import torch
from torch.nn import BYTETracker
from ultralytics import YOLO


def add_keypoints_to_video(video_path, keypoint_extractor, tracker):
    # Abrimos el video
    video = cv2.VideoCapture(video_path)

    # Bucle sobre los frames del video
    while True:
        # Obtenemos el siguiente frame
        ok, frame = video.read()

        # Si el frame no se ha leído correctamente, salimos del bucle
        if not ok:
            break

        # Detectamos los bounding boxes en el frame
        boxes = yolo.predict(frame)

        # Trackeamos los bounding boxes
        tracks = tracker.track(boxes)

        # Obtenemos los keypoints del frame
        keypoints, descriptors = keypoint_extractor.detectAndCompute(frame, None)

        # Agregamos los keypoints a las tracks
        tracks = [add_keypoints(track, keypoints) for track in tracks]

        # Dibujamos las tracks con los keypoints
        draw_tracks_with_keypoints(frame, tracks)

        # Mostramos el frame
        cv2.imshow("Video", frame)
        cv2.waitKey(1)


def add_keypoints(track, keypoints):
    # Convertimos los keypoints a un tensor
    keypoints = torch.tensor(keypoints, dtype=torch.float32)

    # Agregamos los keypoints a la track
    track.keypoints = keypoints

    return track


def draw_tracks_with_keypoints(frame, tracks):
    # Bucle sobre las tracks
    for track in tracks:
        # Obtenemos los bounding box
        bbox = track.box

        # Dibujamos el bounding box
        cv2.rectangle(frame, (bbox.x1, bbox.y1), (bbox.x2, bbox.y2), (0, 255, 0), 2)

        # Dibujamos los keypoints
        for keypoint in track.keypoints:
            cv2.circle(frame, (keypoint[0], keypoint[1]), 5, (255, 0, 0), -1)


if __name__ == "__main__":
    # Cargamos el modelo YOLOv8
    yolo = YOLO()

    # Cargamos el tracker de YOLO
    tracker = BYTETracker()

    # Cargamos el detector de keypoints
    keypoint_extractor = cv2.ORB_create()

    # Ruta del video
    video_path = "video.mp4"

    # Agregamos keypoints a las tracks del video
    add_keypoints_to_video(video_path, keypoint_extractor, tracker)
