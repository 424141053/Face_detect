import cv2
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt5.QtWidgets import QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QWidget
from PyQt5.QtGui import QImage, QPixmap, QPainter, QBrush, QColor
from PyQt5.QtCore import QPoint
from detection.face_detector import FaceDetector


class VideoCaptureThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)

    def __init__(self):
        super().__init__()
        self.cap = cv2.VideoCapture(0)  # 打开摄像头
        self.face_detector = FaceDetector()  # 初始化人脸检测器
        self.frame_with_faces = None  # 初始化帧

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if ret:
                # 使用YOLO进行人脸检测
                self.frame_with_faces = self.face_detector.detect_faces(frame)
                # 发射信号更新UI
                rgb_image = cv2.cvtColor(self.frame_with_faces, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.change_pixmap_signal.emit(q_img)  # 发射信号更新UI


    def stop(self):
        self.cap.release()  # 释放摄像头资源


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("摄像头实时显示")
        self.setGeometry(100, 100, 1000, 600)  # 增加窗口宽度，容纳右侧内容

        # 设置窗口边框颜色为蓝色且加粗5倍
        self.setStyleSheet("QMainWindow { border: 10px solid blue; background-color: #ADD8E6; }")

        # 创建一个主布局，分为左右两个部分
        main_layout = QHBoxLayout()

        # 左边区域 - 显示摄像头数据
        left_layout = QVBoxLayout()

        # 添加提示字样
        self.label_title = QLabel("人脸识别扫描", self)
        self.label_title.setAlignment(Qt.AlignCenter)
        self.label_title.setStyleSheet("font-size: 30px; solid blue; font-weight: bold;")  # 增加字体样式
        left_layout.addWidget(self.label_title)

        # 添加摄像头画面区域
        self.label_video = QLabel(self)
        self.label_video.resize(640, 480)  # 调整摄像头画面大小，避免重合
        left_layout.addWidget(self.label_video)

        # 右边区域 - 显示人员信息
        right_layout = QVBoxLayout()

        # 添加个人信息标签
        self.label_info = QLabel(self)
        self.label_info.setText(
            "姓名: 何研平\n\n"
            "性别: 男\n\n"
            "学号: 2022413210109\n\n"
            "所属学院: 工程师学院\n\n"
            "人员类型: 学生\n\n"
            "入学时间: 2022-9-01\n\n"
            "进入时间: 2024-12-19"
        )
        self.label_info.setStyleSheet("font-size: 25px; font-weight: normal;")  # 设置字体大小和粗细
        right_layout.addWidget(self.label_info, alignment=Qt.AlignBottom)  # 将文本对齐到右下角

        # 将左右区域的布局添加到主布局
        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)

        # 设置布局
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # 创建一个标签显示右上角的图像
        self.label_topright = QLabel(self)
        self.label_topright.setFixedSize(230, 200)  # 设置固定尺寸
        self.label_topright.setStyleSheet("background-color: #FF69B4;")  # 默认粉色背景
        right_layout.addWidget(self.label_topright, alignment=Qt.AlignTop | Qt.AlignRight)  # 添加到右上角

        # 启动线程获取摄像头视频
        self.thread = VideoCaptureThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.start()

        self.frame_to_display = None

    def update_image(self, q_img):
        """更新显示摄像头画面"""
        self.label_video.setPixmap(QPixmap.fromImage(q_img))
        self.add_circle_markers()  # 在显示区域添加圆形标记

    def add_circle_markers(self):
        """在摄像头画面四角绘制圆形"""
        pixmap = self.label_video.pixmap()
        if pixmap is None:
            return  # 如果没有有效的pixmap，不做绘制

        # 使用QPainter绘制圆形
        painter = QPainter(pixmap)
    
        # 定义圆形的大小和位置
        circle_radius = 20
        points = [
            QPoint(0, 0),  # 左上
            QPoint(self.label_video.width() - circle_radius, 0),  # 右上
            QPoint(0, self.label_video.height() - circle_radius),  # 左下
            QPoint(self.label_video.width() - circle_radius, self.label_video.height() - circle_radius)  # 右下
        ]
    
        # 在四个角绘制圆形
        for point in points:
            painter.setBrush(QBrush(Qt.green))  # 设置圆形颜色为绿色
            painter.drawEllipse(point, circle_radius, circle_radius)
    
        painter.end()

    def display_face_frame(self):
        """识别到人脸后，更新为带人脸的帧"""
        # 将frame_with_faces（numpy.ndarray）转换为QImage
        frame_with_faces = self.thread.frame_with_faces
        rgb_image = cv2.cvtColor(frame_with_faces, cv2.COLOR_BGR2RGB)  # 转换为RGB
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # 将处理后的帧显示到右上角
        self.label_topright.setPixmap(QPixmap.fromImage(q_image))
        self.label_topright.setStyleSheet("background-color: #FF69B4;")  # 设置背景为粉色

        # 5秒后恢复背景色并清除图像
        QTimer.singleShot(5000, self.restore_background)

    def restore_background(self):
        """恢复背景为粉色并清空图像"""
        self.label_topright.setStyleSheet("background-color: #FF69B4;")  # 设置为粉色背景
        self.label_topright.clear()  # 清除显示的图像

    def closeEvent(self, event):
        """关闭窗口时释放资源"""
        self.thread.stop()
        event.accept()
