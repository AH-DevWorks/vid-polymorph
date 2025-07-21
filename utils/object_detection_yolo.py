# import cv2
from ultralytics import YOLO
# from matplotlib import pyplot as plt
# import numpy as np
from ultralytics.utils.plotting import Annotator, colors

MODEL = YOLO("files/yolo11m.pt")

def object_detection_yolo(frame):
    results = MODEL.predict(source=frame, conf=0.6, verbose=False)
    res = results[0]

    annotator = Annotator(frame, line_width=2, font_size=14, pil=False)

    boxes = res.boxes
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    clses = boxes.cls.cpu().numpy()

    # 標註所有偵測到的物件
    for i, (box, conf, cls) in enumerate(zip(xyxy, confs, clses)):
        x1, y1, x2, y2 = map(int, box)
        cls = int(cls)
        name = res.names[cls]
        label = f"{name} {conf:.2f}"
        c = colors(cls, bgr=True)  # 為每個類別給不同顏色 :contentReference[oaicite:1]{index=1}
        annotator.box_label([x1, y1, x2, y2], label, color=c)

    # 擷取帶標註的影像
    annotated_frame = annotator.result()
    return annotated_frame