import cv2
import numpy as np
import os
import face_recognition
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt5.QtWidgets import QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QWidget
from PyQt5.QtGui import QImage, QPixmap
from datetime import datetime
from detection.face_detector import FaceRecognizer

class VideoCaptureThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)
    face_frame_signal = pyqtSignal(QImage)

    def __init__(self):
        super().__init__()
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 900)  # 设置宽度
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)  # 设置高度
        self.cap.set(cv2.CAP_PROP_FPS, 30)  # 设置帧率
        self.face_recognizer = FaceRecognizer(tolerance=0.45)  # 调整tolerance提高准确性
        self.running = True
        self.last_recognized_name = None  # 用于跟踪上次识别到的人脸
        self.info_data = {}  # 用于存储从txt文件读取的人员信息
        self.is_info_updated = False  # 用于标记信息是否已更新
        self.unknown_face_shown = False  # 用于标记是否已显示未知人脸

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                # 识别人脸并在帧中标记
                frame_with_faces, recognized_name, recognized_image = self.face_recognizer.recognize_faces(frame.copy())
                rgb_frame = cv2.cvtColor(frame_with_faces, cv2.COLOR_BGR2RGB)

                h, w, ch = rgb_frame.shape
                bytes_per_line = ch * w
                q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.change_pixmap_signal.emit(q_img)

                if recognized_name:
                    # 如果识别到人脸，发送该人脸的框架
                    face_frame = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                    self.face_frame_signal.emit(face_frame)
                    self.face_recognizer.recognized_image = recognized_image  # 保存已知人脸图像路径
                    self.face_recognizer.recognized_name = recognized_name

                    if self.last_recognized_name != recognized_name:
                        self.last_recognized_name = recognized_name  # 更新最后识别到的人脸
                        self.emit_new_face()  # 如果换了人脸，发出新的人脸信号
                        self.is_info_updated = False  # 标记信息未更新

                    # 如果人脸没有变化，且信息尚未更新，更新信息
                    if not self.is_info_updated:
                        self.update_info_text(recognized_name)
                        self.is_info_updated = True  # 更新标志为已更新
                else:
                    # 没有识别到人脸时清空信息和头像
                    self.clear_info_and_image()

    def emit_new_face(self):
        # 发送新的人脸图像
        self.face_frame_signal.emit(QImage(self.face_recognizer.recognized_image))

    def stop(self):
        self.running = False
        self.cap.release()

    def update_info_text(self, recognized_name):
        # 获取当前时间
        current_time = datetime.now().strftime("%Y-%m-%d")

        # 从txt文件读取信息
        if not self.info_data:
            self.load_info_data()

        # 获取该人脸的详细信息
        if recognized_name == "unknow":
            # 如果未匹配到已知人脸，更新为未知信息
            updated_info = f"姓名:未知\n\n" \
                           f"性别: 未知\n\n" \
                           f"学号: 未知\n\n" \
                           f"所属学院: 未知\n\n" \
                           f"人员类型: 未知\n\n" \
                           f"入学时间: 未知\n\n" \
                           f"进入时间: {current_time}"
             # 默认未知人脸图像
            self.face_recognizer.recognized_image = r"face\unknow_face\unknow_face.png" 
            
        elif recognized_name in self.info_data:
            info = self.info_data[recognized_name]
            # 更新信息文本
            updated_info = f"姓名: {info['name']}\n\n" \
                           f"性别: {info['gender']}\n\n" \
                           f"学号: {info['student_id']}\n\n" \
                           f"所属学院: {info['college']}\n\n" \
                           f"人员类型: {info['person_type']}\n\n" \
                           f"入学时间: {info['enrollment_time']}\n\n" \
                           f"进入时间: {current_time}"
            self.face_recognizer.recognized_image = info['image_path']

        self.face_recognizer.recognized_info = updated_info

    def load_info_data(self):
        # 从文件中读取已知人员信息
        txt_path = r"C:\Users\baby\Desktop\大实验\face\information\information.txt"
        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                for line in f:
                    data = line.strip().split(',')
                    if len(data) == 6:  # 确保每行有6个数据项
                        name, gender, student_id, college, person_type, enrollment_time = data
                        self.info_data[name] = {
                            "name": name,
                            "gender": gender,
                            "student_id": student_id,
                            "college": college,
                            "person_type": person_type,
                            "enrollment_time": enrollment_time,
                            "image_path": f"C:/Users/baby/Desktop/大实验/face/known_faces/{name}.jpg"
                        }

    def clear_info_and_image(self):
        # 清除信息和头像
        self.face_recognizer.recognized_name = None
        self.face_recognizer.recognized_info = None
        self.is_info_updated = False  # 重置信息更新标志
        self.unknown_face_shown = False  # 重置未知人脸标志
        # 发送清空信号
        self.face_frame_signal.emit(QImage())  # 发送空的图片，清空显示