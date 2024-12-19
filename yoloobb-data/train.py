from ultralytics import YOLO


def main():
    # model = YOLO('yoloobb-data\\yolov8-obb_seg.yaml',task="obb_seg")
    # model = YOLO('yolov8-obb.yaml').load(
    #     'yoloobb-data\\yolov8n-obb.pt')  # build from YAML and transfer weights

    # # OBB
    # model = YOLO('yolov8-obb.yaml',task="obb")
    #
    # model.train(data='yoloobb-data\\3474data_obb_seg.yaml', epochs=50, imgsz=1024, batch=3*2, workers=6*2, patience=200, )
    # # model.train(data='yoloobb-data\\3474data_obb.yaml', epochs=10, imgsz=1024, batch=3 * 4, workers=6 * 2, patience=2000, )

    # # # SEG
    # model = YOLO('yolov8m-seg.yaml')
    # model.train(data='yoloobb-data\\3474data_obb_seg.yaml', epochs=10, imgsz=1024, batch=2, workers=6, patience=2000, )
    # obb_seg
    model = YOLO('ultralytics\\cfg\\models\\v8\\yolov8-obb_seg.yaml', task="obb_seg")
    # 3474data_obb_seg.yaml
    model.train(data='yoloobb-data\\3474data_obb_seg.yaml', epochs=60, imgsz=1024, batch=4, workers=6*2, patience=200,
                )
    # # #             # hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, degrees=0.0, translate=0.0, scale=0.0, shear=0.0, perspective=0.0,
    # # #             # flipud=0.0, fliplr=0.0, mosaic=0.0, mixup=0.0,
    # # #             # )



if __name__ == '__main__':
    main()