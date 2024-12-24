import cv2
import numpy as np
import os
import face_recognition
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt5.QtWidgets import QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QWidget
from PyQt5.QtGui import QImage, QPixmap
from datetime import datetime

# 人脸识别类
class FaceRecognizer:
    def __init__(self, known_faces_dir=r"C:\Users\baby\Desktop\大实验\face\known_faces", tolerance=0.45):
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_face_images = []
        self.tolerance = tolerance
        self.load_known_faces(known_faces_dir)
        self.recognized_name = None
        self.recognized_image = None
        self.recognized_info = None

    def load_known_faces(self, known_faces_dir):
        if not os.path.exists(known_faces_dir):
            os.makedirs(known_faces_dir)

        for filename in os.listdir(known_faces_dir):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                name = os.path.splitext(filename)[0]
                filepath = os.path.join(known_faces_dir, filename)

                image = face_recognition.load_image_file(filepath)
                encoding = face_recognition.face_encodings(image)
                if encoding:
                    self.known_face_encodings.append(encoding[0])
                    self.known_face_names.append(name)
                    self.known_face_images.append(filepath)  # 保存图片路径
                else:
                    print(f"无法识别 {filename} 中的人脸。")

    def recognize_faces(self, frame):
        # 将帧从 BGR 转换为 RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 在当前帧中检测人脸位置和特征
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        recognized_name = None
        recognized_image = None

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=self.tolerance)
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            name = "未知"
            recognized_name = None

            if matches:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]
                    recognized_name = name
                    recognized_image = self.known_face_images[best_match_index]  # 获取已知人脸图像路径
                else:
                    recognized_name = "unknow"
                    recognized_image = r"face\unknow_face\unknow_face.png" 

            # 在画面中标记人脸
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return frame, recognized_name, recognized_image

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


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("摄像头实时显示")
        self.setGeometry(100, 100, 1000, 600)

        self.setStyleSheet("QMainWindow { border: 10px solid blue; background-color: #ADD8E6; }")

        main_layout = QHBoxLayout()
        left_layout = QVBoxLayout()

        self.label_title = QLabel("人脸识别扫描", self)
        self.label_title.setAlignment(Qt.AlignCenter)
        self.label_title.setStyleSheet("font-size: 30px; solid blue; font-weight: bold;")
        left_layout.addWidget(self.label_title)

        self.label_video = QLabel(self)
        self.label_video.resize(640, 480)
        left_layout.addWidget(self.label_video)

        right_layout = QVBoxLayout()

        # 初始化时，固定文本为空
        self.label_info = QLabel(self)
        self.label_info.setText("")  # 先设为空
        self.label_info.setStyleSheet("font-size: 25px; font-weight: normal;")
        right_layout.addWidget(self.label_info, alignment=Qt.AlignBottom)

        self.label_topright = QLabel(self)
        self.label_topright.setFixedSize(135, 200)  # 设置背景板的固定尺寸
        self.label_topright.setStyleSheet("background-color: #FF69B4;")
        right_layout.addWidget(self.label_topright, alignment=Qt.AlignTop | Qt.AlignRight)

        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        self.thread = VideoCaptureThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.face_frame_signal.connect(self.display_face_frame)
        self.thread.start()

        self.last_recognized_name = None  # 上次识别的人脸

    def update_image(self, q_img):
        self.label_video.setPixmap(QPixmap.fromImage(q_img))

    def display_face_frame(self, face_frame):
        # 如果传递的是空的图片，清空头像显示
        if face_frame.isNull():
            self.label_topright.clear()  # 清空头像
            self.label_info.setText("")  # 清空信息文本
        else:
            # 获取粉红色背景板的尺寸
            background_size = self.label_topright.size()

            # 缩放人脸图片以适应背景板的大小
            scaled_face_frame = face_frame.scaled(background_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)

            # 获取当前识别的人脸名称
            current_recognized_name = self.thread.face_recognizer.recognized_name

            if current_recognized_name:
                # 更新头像
                known_face_pixmap = QPixmap(self.thread.face_recognizer.recognized_image)
                scaled_known_face = known_face_pixmap.scaled(background_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.label_topright.setPixmap(scaled_known_face)

                # 更新信息文本
                self.label_info.setText(self.thread.face_recognizer.recognized_info)
            else:
                # 没有识别到人脸时，清空信息和头像
                self.label_topright.clear()
                self.label_info.setText("")  # 清空文本

    def closeEvent(self, event):
        self.thread.stop()
        self.thread.wait()
        event.accept()

if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication
    import sys

    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
