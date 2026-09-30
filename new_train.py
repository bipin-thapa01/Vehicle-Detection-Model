#i switched to a new pc with better cpu and gpu 
#so this file is for resuming training but in new device

from ultralytics import YOLO

def main():
    model = YOLO("runs/detect/train-4/weights/last.pt")

    model.train(
        data="dataset/data.yaml",
        epochs=100,
        imgsz=640,
        batch=8,
        workers=8,
        device=0,
        resume=True,
    )

if __name__ == "__main__":
    main()