from PyQt5.QtWidgets import QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QWidget
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from utils.video_capture import VideoCaptureThread

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