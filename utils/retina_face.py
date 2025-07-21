# 人臉偵測 + 馬賽克功能 - RetinaFace + Mosaic
import cv2
from retinaface import RetinaFace

FONT = cv2.FONT_HERSHEY_DUPLEX
LINE_TYPE = cv2.LINE_AA

def retina_face(frame, mosaic=False):
    """
    使用 RetinaFace 模型檢測人臉，並可選擇是否對人臉進行馬賽克處理
    :param input_frame: 輸入的影像幀
    :param mosaic: 是否啟用馬賽克（預設 False）
    :return: 偵測完畢並標記人臉及臉部特徵點的影像幀
    """
    faces = RetinaFace.detect_faces(frame)
    
    # 臉部輪廓偵測
    for _, face in faces.items():
        x1, y1, x2, y2 = face["facial_area"]

        # Default：畫臉框及信心水準
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0),2)
        cv2.putText(
            img=frame,
            text=f"{face['score']:.3f}",
            org=(x1, y1),
            fontFace=FONT,
            fontScale=0.9,
            color=(30, 230, 110),
            thickness=2,
            lineType=LINE_TYPE
        )

        if mosaic:
            # 🎯 最小修改：馬賽克區塊
            face_roi = frame[y1:y2, x1:x2]
            if face_roi.size > 0:
                small = cv2.resize(face_roi, (10, 10), interpolation=cv2.INTER_LINEAR)
                mosaic_block = cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
                frame[y1:y2, x1:x2] = mosaic_block
        else:
            # 標出人臉特徵點（眼睛、鼻子、嘴巴）
            lm = face['landmarks']
            for part in ["right_eye", "left_eye", "nose", "mouth_right", "mouth_left"]:
                x, y = int(lm[part][0]), int(lm[part][1])
                cv2.circle(frame, (x, y), 2, (6, 164, 125), -1)
    
    return frame