import cv2
import numpy as np

def masking(frame: np.ndarray, mask_choice: str) -> np.ndarray:
    """
    將 mask 應用到 frame 上，返回遮罩後的影像。
    :param frame: 原始影像
    :param mask: 二值遮罩影像
    :return: 遮罩後的影像
    """
    mask_choice = mask_choice.strip().lower() if isinstance(mask_choice, str) else "circle"
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)

    if mask_choice == "circle":
        cv2.circle(mask, (frame.shape[1] // 2, frame.shape[0] // 2), min(frame.shape[:2]) // 3, 255, -1)
    elif mask_choice == "rectangle":
        cv2.rectangle(mask, (frame.shape[1] // 5, frame.shape[0] // 5),
                      (4 * frame.shape[1] // 5, 4 * frame.shape[0] // 5), 255, -1)
    elif mask_choice == "ellipse":
        cv2.ellipse(mask, (frame.shape[1] // 2, frame.shape[0] // 2),
                    (frame.shape[1] // 3, frame.shape[0] // 3), 0, 0, 360, 255, -1)
    else:
        # triangle mask
        pts = np.array(((10, frame.shape[0] // 4 * 3),
                (frame.shape[1] - 10, frame.shape[0] // 4 * 3),
                (frame.shape[1] // 2, 10)))
        cv2.fillPoly(mask, [pts], 255)
    
    masked = cv2.bitwise_and(frame, frame, mask=mask)
    cv2.putText(
        masked, f"{mask_choice.capitalize()}",
        (15, 50),
        cv2.FONT_HERSHEY_SIMPLEX, 1.35, (0,255,255), 3, cv2.LINE_AA
    )

    return masked