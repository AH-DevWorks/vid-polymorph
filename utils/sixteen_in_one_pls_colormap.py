# 16 in 1 + Colormap
import cv2
import numpy as np
import random

# 所有 colormap 效果封裝
colormap_effects = [
    lambda x: cv2.applyColorMap(x, cv2.COLORMAP_AUTUMN),
    lambda x: cv2.applyColorMap(x, cv2.COLORMAP_BONE),
    lambda x: cv2.applyColorMap(x, cv2.COLORMAP_JET),
    lambda x: cv2.applyColorMap(x, cv2.COLORMAP_WINTER),
    lambda x: cv2.applyColorMap(x, cv2.COLORMAP_RAINBOW),
    lambda x: cv2.applyColorMap(x, cv2.COLORMAP_OCEAN),
    lambda x: cv2.applyColorMap(x, cv2.COLORMAP_SUMMER),
    lambda x: cv2.applyColorMap(x, cv2.COLORMAP_SPRING),
    lambda x: cv2.applyColorMap(x, cv2.COLORMAP_COOL),
    lambda x: cv2.applyColorMap(x, cv2.COLORMAP_HSV),
    lambda x: cv2.applyColorMap(x, cv2.COLORMAP_PINK),
    lambda x: cv2.applyColorMap(x, cv2.COLORMAP_HOT),
    lambda x: cv2.applyColorMap(x, cv2.COLORMAP_PARULA),
    lambda x: cv2.applyColorMap(x, cv2.COLORMAP_PLASMA),
    lambda x: cv2.applyColorMap(x, cv2.COLORMAP_INFERNO),
    lambda x: cv2.applyColorMap(x, cv2.COLORMAP_VIRIDIS),
]

colormap_names = [
    "AUTUMN", "BONE", "JET", "WINTER", "RAINBOW", "OCEAN", "SUMMER", "SPRING",
    "COOL", "HSV", "PINK", "HOT", "PARULA", "PLASMA", "INFERNO", "VIRIDIS"
]

# 初始化位置順序
position_order = list(range(16))

def sixteen_in_one_pls_colormap(base_frame: np.ndarray, original_frame: np.ndarray,
                frame_idx: int, position_order: list, effects: list = colormap_effects) -> np.ndarray:
    h, w = base_frame.shape[:2]
    small = cv2.resize(original_frame, (w//4, h//4), interpolation=cv2.INTER_AREA)

    start_frame = 1010
    if (frame_idx - start_frame) % 50 == 0:
        position_order = random.sample(position_order, len(position_order))

    for idx in range(16):
        i, j = divmod(position_order[idx], 4)  # 轉換成畫面上的座標格
        img = effects[idx](small.copy()) if effects and idx < len(effects) else small
        
        # 上字 - 效果名稱
        label = colormap_names[idx] if idx < len(colormap_names) else "UNKNOWN"
        cv2.putText(img, label, (10, img.shape[0] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)

        x, y = j * w//4, i * h//4
        base_frame[y:y + h//4, x:x + w//4] = img

    return base_frame, position_order