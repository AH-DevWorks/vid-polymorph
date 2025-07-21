# Edge Detection (Sobel / Canny / Laplacian)
import cv2
import numpy as np

def apply_edge_detection_by_region(frame):
    h, w = frame.shape[:2]
    third_w = w // 3

    # 分割成三塊（left, mid, right）
    left = frame[:, :third_w]
    mid = frame[:, third_w:2*third_w]
    right = frame[:, 2*third_w:]

    # 左側：Sobel
    sobel_x = cv2.Sobel(cv2.cvtColor(left, cv2.COLOR_BGR2GRAY), cv2.CV_64F, 1, 0)
    sobel_y = cv2.Sobel(cv2.cvtColor(left, cv2.COLOR_BGR2GRAY), cv2.CV_64F, 0, 1)
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    magnitude = cv2.convertScaleAbs(magnitude)
    left_result = cv2.cvtColor(magnitude, cv2.COLOR_GRAY2BGR)

    # 中間：Canny
    # 從 sobel 取梯度強度來決定 threshold
    max_grad = np.max(magnitude)
    high_thresh = int(0.35 * max_grad)
    low_thresh = int(0.15 * max_grad)
    canny = cv2.Canny(cv2.cvtColor(mid, cv2.COLOR_BGR2GRAY), low_thresh, high_thresh)
    mid_result = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)

    # 右側：Laplacian
    laplacian = cv2.Laplacian(cv2.cvtColor(right, cv2.COLOR_BGR2GRAY), cv2.CV_64F)
    laplacian = cv2.convertScaleAbs(laplacian)
    right_result = cv2.cvtColor(laplacian, cv2.COLOR_GRAY2BGR)

    # 加文字
    cv2.putText(left_result, "Sobel", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.35, (0,255,255), 3)
    cv2.putText(mid_result, "Canny", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.35, (0,255,255), 3)
    cv2.putText(right_result, "Laplacian", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.35, (0,255,255), 3)

    # 合併
    combined = np.hstack([left_result, mid_result, right_result])
    return combined
