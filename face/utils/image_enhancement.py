import cv2
import numpy as np

def enhance_image(frame):
    """对图像应用去噪与锐化处理"""
    # 高斯模糊去噪
    smoothed_frame = cv2.GaussianBlur(frame, (5, 5), 0)

    # 锐化处理（使用卷积核）
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])  # 锐化核
    sharpened_frame = cv2.filter2D(smoothed_frame, -1, kernel)

    return sharpened_frame
