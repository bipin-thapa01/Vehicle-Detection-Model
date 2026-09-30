from ultralytics import YOLO

model = YOLO("runs/detect/train-4/weights/last.pt")

model.train(
    resume=True,
)