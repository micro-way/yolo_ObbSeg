from ultralytics import YOLO


def main():
    model = YOLO(r'runs/obb/train9/weights/best.pt')
    model.val(data='my-dota8-obb.yaml', imgsz=640, batch=4, workers=4)


if __name__ == '__main__':
    main()
    