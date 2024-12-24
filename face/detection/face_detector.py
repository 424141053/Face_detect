import cv2
import numpy as np
import os
import face_recognition
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
