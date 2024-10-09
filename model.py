from ultralytics import YOLO


if __name__ == "__main__":
    model = YOLO("yolov8x.pt")
    model.train(data='custom.yaml', epochs=100, batch=16)