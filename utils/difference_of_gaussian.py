# DoG
import numpy as np
import cv2
def difference_of_gaussian(frame, ksize=(5,5), sigma_a=1.0, sigma_b=1.6):
    blur_1 = cv2.GaussianBlur(frame.astype(np.float32), ksize, sigma_a)
    blur_2 = cv2.GaussianBlur(frame.astype(np.float32), ksize, sigma_b)
    frame_dog = blur_1 - blur_2
    res_frame = cv2.convertScaleAbs(frame_dog, alpha=25.0)
    return res_frame