# Morphological Operations (Dilation / Opening / Gradient)
import cv2
import numpy as np

def morphological_operation_by_region(frame):
    h, w = frame.shape[:2]
    third_w = w // 3

    # 分割成三塊（left, mid, right）
    left = frame[:, :third_w]
    mid = frame[:, third_w:2*third_w]
    right = frame[:, 2*third_w:]

    # 左側：Gradient
    kernel_gd = np.ones((5, 5), np.uint8)
    left_result = cv2.morphologyEx(left, cv2.MORPH_GRADIENT, kernel_gd)

    # 中間：Opening
    kernel_op = np.ones((5, 5), np.uint8)
    mid_result = cv2.morphologyEx(mid, cv2.MORPH_OPEN, kernel_op)

    # 右側：Dilation
    kernel_77 = np.ones((5, 5), np.uint8)
    right_result = cv2.dilate(right, kernel_77)
    # b, g, r = cv2.split(left)
    # b_d = cv2.dilate(b, kernel_77)
    # g_d = cv2.dilate(g, kernel_77)
    # r_d = cv2.dilate(r, kernel_77)
    # left_result = cv2.merge([b_d, g_d, r_d])

    # 加文字
    cv2.putText(left_result, "Gradient", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.35, (0,255,255), 3)
    cv2.putText(mid_result, "Opening", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.35, (0,255,255), 3)
    cv2.putText(right_result, "Dilation", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.35, (0,255,255), 3)

    # 合併
    combined = np.hstack([left_result, mid_result, right_result])
    return combined