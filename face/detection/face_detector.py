import cv2
from ultralytics import YOLO
import numpy as np

class FaceDetector:
    def __init__(self):
        # 加载YOLOv11模型（假设你使用的是YOLOv11的best.pt文件）
        self.yolo_model = YOLO(r"C:\Users\baby\runs\detect\train3\weights\best.pt")
        self.class_names = self.yolo_model.names  # 获取类别名称

    def detect_faces(self, frame):
        # YOLO模型推理
        # 将BGR图像转换为RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.yolo_model.predict(rgb_frame, conf=0.01, augment=False)

        # 解析结果
        boxes = results[0].boxes
        xywh = boxes.xywh.numpy().astype(int)  # 获取坐标信息
        cls = boxes.cls.numpy().astype(int)   # 获取类别信息
        score = boxes.conf.numpy().astype(float)  # 获取置信度信息

        # 绘制矩形框和标签
        for i in range(len(xywh)):
            x_center, y_center, w, h = xywh[i]
            x1 = int(x_center - w / 2)
            y1 = int(y_center - h / 2)
            x2 = int(x_center + w / 2)
            y2 = int(y_center + h / 2)

            # 绘制边界框
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 获取类别名称
            label = self.class_names[cls[i]]

            # 绘制类别名称和置信度
            text = f"{label} {score[i]:.2f}"
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return frame
