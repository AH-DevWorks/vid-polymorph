# Feature Detection & Descriptor (SIFT / SuperPoint / ORB)
import cv2
import torch
import numpy as np
from superpoint_pytorch import SuperPoint

def feature_detection_and_descriptor(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = frame.shape[:2]
    third_w = w // 3

    # 分割成三塊（left, mid, right）
    left = frame[:, :third_w]
    mid = frame[:, third_w:2*third_w]
    right = frame[:, 2*third_w:]

    # 左側：Scale-Invariant Feature Transform (SIFT)
    sift = cv2.SIFT_create()
    kp_sift = sift.detect(left, None)
    left_result = cv2.drawKeypoints(left, kp_sift, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # 中間：SuperPoint
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load("files/superpoint_v6_from_tf.pth", map_location=device)
    model = SuperPoint().to(device)
    model.load_state_dict(state_dict)
    model.eval()

    img = mid.astype(np.float32) / 255.0
    inp = torch.from_numpy(img[None, None]).float().to(device)

    # 推論
    with torch.no_grad():
        out = model({'image': inp})
    pts = out['keypoints'][0].detach().cpu().numpy()       # 特徵點 (N×2)
    desc = out['descriptors'][0].detach().cpu().numpy()    # 描述子 (D×N)

    # 畫出特徵點
    img_vis = cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    for x, y in pts:
        cv2.circle(img_vis, (int(x), int(y)), 2, (0,255,0), -1)
    
    mid_result = img_vis

    # 右側：Oriented FAST and rotated BRIEF (ORB)
    orb_feature = cv2.ORB_create()
    orb_kp  = orb_feature.detect(right)
    orb_out  = cv2.drawKeypoints(right, orb_kp, None)
    right_result = orb_out

    # 加文字
    cv2.putText(left_result, "SIFT", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.35, (0,255,255), 3)
    cv2.putText(mid_result, "SuperPoint", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.35, (0,255,255), 3)
    cv2.putText(right_result, "ORB", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.35, (0,255,255), 3)

    # 合併
    combined = np.hstack([left_result, mid_result, right_result])
    return combined