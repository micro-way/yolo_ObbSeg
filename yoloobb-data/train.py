from ultralytics import YOLO


def main():
    # model = YOLO('yoloobb-data\\yolov8-obb_seg.yaml',task="obb_seg")
    # model = YOLO('yolov8-obb.yaml').load(
    #     'yoloobb-data\\yolov8n-obb.pt')  # build from YAML and transfer weights

    # # OBB
    # model = YOLO('yolov8-obb.yaml',task="obb")
    # model.train(data='yoloobb-data\\data_obb.yaml', epochs=20, imgsz=1024, batch=2, workers=4, patience=2000, )

    # # # SEG
    # model = YOLO('yolov8-seg.yaml')
    # model.train(data='yoloobb-data\\data_seg.yaml', epochs=20, imgsz=1024, batch=2, workers=4, patience=2000, )

    # obb_seg
    model = YOLO('yolov8-obb_seg_MyCBAM.yaml',task="obb_seg")
    model.train(data='yoloobb-data\\data_obb_seg.yaml', epochs=20, imgsz=1024, batch=2, workers=4, patience=2000, )



if __name__ == '__main__':
    main()