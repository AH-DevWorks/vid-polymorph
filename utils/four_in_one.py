import cv2
import numpy as np
from .overlay_original_preview import overlay_original_preview

def four_in_one(base_frame: np.ndarray, original_frame: np.ndarray,
                effects: list = None) -> np.ndarray:
    """
    將影像分割成四個等比例縮小區塊，並返回四個處理後同步播放的影像。
    每格可以套不同效果（effects 是函數 list）。
    """
    h, w = base_frame.shape[:2]
    pframe_positions = [
        (0, 0), (w//2, 0), (0, h//2), (w//2, h//2)
    ]
    small = cv2.resize(original_frame, (w//2, h//2), interpolation=cv2.INTER_AREA)
    
    for i in range(4):
        img = small.copy()
        if effects and effects[i]:
            img = effects[i](img)
        overlay_original_preview(
            base_frame, img,
            position=pframe_positions[i],
            scale=1.0, border_thickness=0
        )
    return base_frame