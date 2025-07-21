import cv2
import numpy as np

def overlay_original_preview(
        base_frame: np.ndarray,
        original_frame: np.ndarray,
        position: tuple = None,
        scale: float = 0.25,
        border_color: tuple = (0, 59, 133),
        border_thickness: int = 1
) -> np.ndarray:
    """
    將縮小後的 original_frame 疊加到基底影像 base_frame 上。
     - position: tuple (x, y) - 縮小後的影像在 base_frame 上的貼齊位置。
        若為 None，則使用預設貼齊左側的垂直中偏上位置 (x=0, y=base_h // 4)。
     - scale: 縮小比例。
     - border_thickness: 外框線粗細。
     - border_color: 外框線顏色，預設為 (0, 59, 133)。
    """
    base_h, base_w = base_frame.shape[:2]
    orig_h, orig_w = original_frame.shape[:2]

    small_w, small_h = int(orig_w * scale), int(orig_h * scale)

    # 縮圖後疊加
    resized = cv2.resize(original_frame, (small_w, small_h), interpolation=cv2.INTER_AREA)

    # 動態定位位置
    if position is None:
        x = 0
        y = base_h // 4
    else:
        x, y = position
        # 避免越界
        x = max(0, min(x, base_w - small_w))
        y = max(0, min(y, base_h - small_h))

    base_frame[y:y+small_h, x:x+small_w] = resized
    # 畫細邊框
    cv2.rectangle(base_frame, (x, y), (x+small_w, y+small_h), border_color, border_thickness)

    return base_frame