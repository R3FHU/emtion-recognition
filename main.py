import sys
import cv2
import time
import os
import torch
import librosa
import numpy as np
import joblib
import pyaudio
from PIL import Image

from PyQt6.QtWidgets import (
    QApplication, QLabel, QPushButton, QVBoxLayout, QWidget,
    QHBoxLayout, QFileDialog, QFrame, QProgressBar
)
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtGui import QImage, QPixmap

from ultralytics import YOLO
from transformers import AutoImageProcessor, AutoModelForImageClassification

# 引入我们自定义的热力图组件
from heatmap_widget import EmotionHeatmapWidget

# 获取绝对路径
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_PATH, "models")

# 统一全局情绪标签顺序（用于对齐不同模型的输出）
CANONICAL_EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']


# =========================
# 1. 音频识别基模型结构
# =========================
class AudioMLP(torch.nn.Module):
    def __init__(self, input_dim=120, num_classes=7):
        super(AudioMLP, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.4),
            torch.nn.Linear(512, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, num_classes)
        )

    def forward(self, x, return_features=False):
        features = x
        for layer in list(self.model.children())[:-1]:
            features = layer(features)
        out = list(self.model.children())[-1](features)

        if return_features:
            return out, features
        return out


# =========================
# 1.5 中期跨模态特征融合网络
# =========================
class CrossModalFusionNet(torch.nn.Module):
    def __init__(self, input_dim=896, num_classes=7):
        super(CrossModalFusionNet, self).__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.4),
            torch.nn.Linear(256, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.network(x)


# =========================
# 2. 音频处理线程
# =========================
class AudioThread(QThread):
    audio_prob_signal = pyqtSignal(np.ndarray, float, np.ndarray)

    def __init__(self, device):
        super().__init__()
        self.device = device
        self.running = False

        try:
            self.scaler = joblib.load(os.path.join(MODELS_DIR, 'audio_scaler.pkl'))
            self.le = joblib.load(os.path.join(MODELS_DIR, 'label_encoder.pkl'))
            self.model = AudioMLP(input_dim=120, num_classes=len(self.le.classes_)).to(device)
            self.model.load_state_dict(torch.load(os.path.join(MODELS_DIR, 'best_audio_model.pt'), map_location=device))
            self.model.eval()

            self.mapping_idx = [list(self.le.classes_).index(emo) if emo in self.le.classes_ else -1
                                for emo in CANONICAL_EMOTIONS]
        except Exception as e:
            print(f"音频组件加载失败: {e}")

        self.p = pyaudio.PyAudio()

    def run(self):
        self.running = True
        stream = self.p.open(format=pyaudio.paFloat32, channels=1, rate=22050, input=True, frames_per_buffer=1024)
        frames = []
        while self.running:
            try:
                data = stream.read(1024, exception_on_overflow=False)
                frames.append(np.frombuffer(data, dtype=np.float32))
                if len(frames) >= 54:
                    audio_data = np.concatenate(frames)
                    frames = []
                    mfccs = librosa.feature.mfcc(y=audio_data, sr=22050, n_mfcc=40)
                    feat = np.hstack((np.mean(mfccs.T, axis=0),
                                      np.mean(librosa.feature.delta(mfccs).T, axis=0),
                                      np.mean(librosa.feature.delta(mfccs, order=2).T, axis=0)))
                    feat_scaled = self.scaler.transform(feat.reshape(1, -1))

                    with torch.no_grad():
                        output, audio_feat = self.model(torch.FloatTensor(feat_scaled).to(self.device),
                                                        return_features=True)
                        prob = torch.nn.functional.softmax(output, dim=1).cpu().numpy()[0]
                        a_feat_np = audio_feat.cpu().numpy()[0]

                    aligned_prob = np.array([prob[i] if i != -1 else 0.0 for i in self.mapping_idx])
                    self.audio_prob_signal.emit(aligned_prob, float(np.max(aligned_prob)), a_feat_np)

            except:
                continue

        stream.stop_stream()
        stream.close()

    def stop(self):
        self.running = False
        self.wait()


# =========================
# 3. 视觉处理线程 (已修复摄像头启动卡死)
# =========================
class VideoThread(QThread):
    frame_signal = pyqtSignal(object)
    visual_prob_signal = pyqtSignal(np.ndarray, float, np.ndarray)

    def __init__(self, face_model, emotion_model, processor, device):
        super().__init__()
        self.face_model = face_model
        self.emotion_model = emotion_model
        self.processor = processor
        self.device = device
        self.running = False
        self.cap = None
        self.source = None

        self.id2label = self.emotion_model.config.id2label
        self.mapping_idx = [next((k for k, v in self.id2label.items() if v.lower() == emo), -1)
                            for emo in CANONICAL_EMOTIONS]

    def start_camera(self):
        self.source = 0
        self.running = True
        self.start()  # 仅启动线程，把打开摄像头的耗时操作抛给 run() 函数

    def start_video(self, path):
        self.source = path
        self.running = True
        self.start()

    def run(self):
        # [防卡死修复] 在后台线程初始化 VideoCapture
        if self.source == 0:
            if os.name == 'nt':
                # Windows 环境加上 cv2.CAP_DSHOW 绕过 MSMF 的延迟
                self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            else:
                self.cap = cv2.VideoCapture(0)
        else:
            self.cap = cv2.VideoCapture(self.source)

        if not self.cap or not self.cap.isOpened():
            print("❌ 无法打开摄像头或视频文件")
            self.running = False
            return

        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret: break

            results = self.face_model.predict(frame, device=self.device, imgsz=320, conf=0.4, verbose=False)
            v_prob = np.zeros(len(CANONICAL_EMOTIONS))
            v_quality = 0.0
            v_feat_np = np.zeros(768)

            if results and results[0].boxes:
                box = results[0].boxes[0]
                v_quality = float(box.conf[0].item())

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                face = frame[max(0, y1):y2, max(0, x1):x2]
                if face.size != 0:
                    pil_img = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
                    inputs = self.processor(images=pil_img, return_tensors="pt").to(self.device)
                    with torch.no_grad():
                        outputs = self.emotion_model(**inputs, output_hidden_states=True)
                        prob = torch.nn.functional.softmax(outputs.logits, dim=1).cpu().numpy()[0]
                        v_feat_np = outputs.hidden_states[-1][:, 0, :].cpu().numpy()[0]

                    v_prob = np.array([prob[i] if i != -1 else 0.0 for i in self.mapping_idx])

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (26, 115, 232), 2)
                    top_emo = CANONICAL_EMOTIONS[np.argmax(v_prob)]
                    cv2.putText(frame, top_emo, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (26, 115, 232), 2)

            self.frame_signal.emit(frame)
            self.visual_prob_signal.emit(v_prob, v_quality, v_feat_np)

            # [防卡死修复] 释放极小部分时间片给系统事件队列
            time.sleep(0.01)

    def stop(self):
        self.running = False
        self.wait()
        if self.cap: self.cap.release()


# =========================
# 4. Modern App UI (双轨融合系统)
# =========================
class ModernApp(QWidget):
    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.fusion_mode = "LATE"  # 默认为后期动态加权融合
        self.last_v_feat = np.zeros(768)
        self.last_a_feat = np.zeros(128)
        self.last_v_prob = np.zeros(len(CANONICAL_EMOTIONS))
        self.last_a_prob = np.zeros(len(CANONICAL_EMOTIONS))
        self.v_weight = 0.5
        self.a_weight = 0.5

        self.init_models()
        self.setup_ui()
        self.setup_threads()

    def init_models(self):
        # 1. 视觉基模型加载
        yolo_path = os.path.join(MODELS_DIR, "runs/detect/runs/face/yolo11n_maxperf/weights/best.pt")
        self.face_model = YOLO(yolo_path)

        vit_local_path = os.path.join(MODELS_DIR, "vit-face-expression").replace("\\", "/")
        self.processor = AutoImageProcessor.from_pretrained(vit_local_path)
        self.emotion_model = AutoModelForImageClassification.from_pretrained(vit_local_path).to(self.device)

        # 2. 中期融合模型和 Scaler 加载
        try:
            self.fusion_scaler = joblib.load(os.path.join(MODELS_DIR, 'mid_fusion_scaler.pkl'))
            self.fusion_encoder = joblib.load(os.path.join(MODELS_DIR, 'mid_fusion_encoder.pkl'))

            # [防死锁修复] 强制在中期融合网络在 CPU 上运行，以防在主线程与视频线程抢夺 GPU 引起死锁
            self.mid_fusion_model = CrossModalFusionNet(input_dim=896, num_classes=7).to("cpu")
            self.mid_fusion_model.load_state_dict(
                torch.load(os.path.join(MODELS_DIR, 'best_mid_fusion_model.pt'), map_location="cpu"))
            self.mid_fusion_model.eval()
            print("✅ 中期跨模态融合神经网络及 Scaler 加载成功！")
        except Exception as e:
            print(f"❌ 中期融合网络加载失败: {e}")

    def setup_ui(self):
        self.setWindowTitle("AI 多模态情绪分析系统 - 动态融合版")
        self.setFixedSize(1150, 850)
        self.setStyleSheet("""
            QWidget { background-color: #F1F3F4; font-family: 'Segoe UI', Roboto; }
            #Card { background-color: white; border-radius: 12px; border: 1px solid #DADCE0; }
            #Title { color: #1A73E8; font-size: 26px; font-weight: 600; padding: 10px; }
            #LabelHeader { color: #5F6368; font-size: 13px; font-weight: bold; text-transform: uppercase; }
            #EmotionResult { color: #202124; font-size: 24px; font-weight: bold; }
            #FusionResult { color: #1A73E8; font-size: 32px; font-weight: 800; }
            QPushButton { background-color: #1A73E8; color: white; border-radius: 6px; padding: 12px 20px; font-weight: bold; }
            QPushButton:hover { background-color: #1765CC; }
            QPushButton#SecondaryBtn { background-color: white; color: #1A73E8; border: 1px solid #DADCE0; }
            QPushButton#StopBtn { background-color: white; color: #D93025; border: 1px solid #DADCE0; }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 20, 30, 30)

        header = QHBoxLayout()
        title = QLabel("Multimodal Emotion Intelligence")
        title.setObjectName("Title")
        header.addWidget(title)
        self.fusion_display = QLabel("READY")
        self.fusion_display.setObjectName("FusionResult")
        header.addStretch()
        header.addWidget(self.fusion_display)
        layout.addLayout(header)

        content = QHBoxLayout()
        self.video_card = QFrame(objectName="Card")
        v_layout = QVBoxLayout(self.video_card)
        self.video_view = QLabel("等待输入源...")
        self.video_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_view.setFixedSize(720, 540)
        self.video_view.setStyleSheet("background-color: #202124; border-radius: 8px;")
        v_layout.addWidget(self.video_view)
        content.addWidget(self.video_card)

        stats = QVBoxLayout()
        v_card = QFrame(objectName="Card")
        vl = QVBoxLayout(v_card)
        vl.addWidget(QLabel("Visual Analysis", objectName="LabelHeader"))
        self.v_res = QLabel("IDLE", objectName="EmotionResult")
        self.v_weight_label = QLabel("Weight: 0.00")
        vl.addWidget(self.v_res)
        vl.addWidget(self.v_weight_label)
        stats.addWidget(v_card)

        a_card = QFrame(objectName="Card")
        al = QVBoxLayout(a_card)
        al.addWidget(QLabel("Audio Analysis", objectName="LabelHeader"))
        self.a_res = QLabel("IDLE", objectName="EmotionResult")
        self.a_bar = QProgressBar()
        self.a_weight_label = QLabel("Weight: 0.00")
        al.addWidget(self.a_res)
        al.addWidget(self.a_bar)
        al.addWidget(self.a_weight_label)
        stats.addWidget(a_card)

        heatmap_card = QFrame(objectName="Card")
        hl = QVBoxLayout(heatmap_card)
        hl.addWidget(QLabel("Fusion Heatmap", objectName="LabelHeader"))

        self.dynamic_heatmap = EmotionHeatmapWidget(history_length=50)
        self.dynamic_heatmap.setMinimumSize(250, 160)

        hl.addWidget(self.dynamic_heatmap)
        stats.addWidget(heatmap_card)

        stats.addStretch()
        content.addLayout(stats)
        layout.addLayout(content)

        footer = QHBoxLayout()
        self.btn_cam = QPushButton("启动系统 (Cam)")
        self.btn_file = QPushButton("检测视频文件", objectName="SecondaryBtn")

        self.btn_fusion_toggle = QPushButton("当前: 动态后期融合 (Late)")
        self.btn_fusion_toggle.setStyleSheet("background-color: #34A853; color: white;")

        self.btn_stop = QPushButton("停止系统", objectName="StopBtn")

        footer.addWidget(self.btn_cam)
        footer.addWidget(self.btn_file)
        footer.addWidget(self.btn_fusion_toggle)
        footer.addStretch()
        footer.addWidget(self.btn_stop)
        layout.addLayout(footer)

    def setup_threads(self):
        self.video_thread = VideoThread(self.face_model, self.emotion_model, self.processor, self.device)
        self.audio_thread = AudioThread(self.device)
        self.video_thread.frame_signal.connect(self.update_video_frame)
        self.video_thread.visual_prob_signal.connect(self.process_visual_logic)
        self.audio_thread.audio_prob_signal.connect(self.process_audio_logic)
        self.btn_cam.clicked.connect(self.start_all_cam)
        self.btn_file.clicked.connect(self.open_file)
        self.btn_stop.clicked.connect(self.stop_all)
        self.btn_fusion_toggle.clicked.connect(self.toggle_fusion_mode)

    def toggle_fusion_mode(self):
        if self.fusion_mode == "LATE":
            self.fusion_mode = "MID"
            self.btn_fusion_toggle.setText("当前: 特征中期融合 (Mid)")
            self.btn_fusion_toggle.setStyleSheet("background-color: #EA4335; color: white;")
            self.fusion_display.setStyleSheet("color: #8E24AA; font-size: 32px; font-weight: 800;")
        else:
            self.fusion_mode = "LATE"
            self.btn_fusion_toggle.setText("当前: 动态后期融合 (Late)")
            self.btn_fusion_toggle.setStyleSheet("background-color: #34A853; color: white;")
            self.fusion_display.setStyleSheet("color: #1A73E8; font-size: 32px; font-weight: 800;")

    def start_all_cam(self):
        self.video_thread.start_camera()
        self.audio_thread.start()

    def process_visual_logic(self, prob_vec, quality, feat_vec):
        self.last_v_prob = prob_vec
        self.v_weight = quality * np.max(prob_vec) if np.any(prob_vec) else 0.0
        self.last_v_feat = feat_vec
        top_emo = CANONICAL_EMOTIONS[np.argmax(prob_vec)] if np.any(prob_vec) else "NONE"
        self.v_res.setText(top_emo.upper())
        self.perform_fusion()

    def process_audio_logic(self, prob_vec, confidence, feat_vec):
        self.last_a_prob = prob_vec
        self.a_weight = confidence
        self.last_a_feat = feat_vec
        top_emo = CANONICAL_EMOTIONS[np.argmax(prob_vec)]
        self.a_res.setText(top_emo.upper())
        self.a_bar.setValue(int(confidence * 100))
        self.perform_fusion()

    def perform_fusion(self):
        if not np.any(self.last_a_feat) or not np.any(self.last_v_feat):
            return

        if self.fusion_mode == "LATE":
            total_w = self.v_weight + self.a_weight + 1e-6
            norm_v_w = self.v_weight / total_w
            norm_a_w = self.a_weight / total_w
            fused_prob = (norm_v_w * self.last_v_prob) + (norm_a_w * self.last_a_prob)
            self.v_weight_label.setText(f"Late Weight: {norm_v_w:.2f}")
            self.a_weight_label.setText(f"Late Weight: {norm_a_w:.2f}")

        elif self.fusion_mode == "MID":
            try:
                fused_feat = np.hstack((self.last_a_feat, self.last_v_feat))
                fused_scaled = self.fusion_scaler.transform(fused_feat.reshape(1, -1))

                with torch.no_grad():
                    # [防死锁修复] 在 CPU 上执行推理
                    tensor_in = torch.FloatTensor(fused_scaled).to("cpu")
                    out = self.mid_fusion_model(tensor_in)

                    prob = torch.nn.functional.softmax(out, dim=1).numpy()[0]
                    fused_prob = prob

                self.v_weight_label.setText("Mid Fusion: Active (768d)")
                self.a_weight_label.setText("Mid Fusion: Active (128d)")
            except Exception as e:
                print(f"融合推理出错: {e}")
                return

        final_idx = np.argmax(fused_prob)
        final_emotion = CANONICAL_EMOTIONS[final_idx]
        final_conf = fused_prob[final_idx]

        if hasattr(self, 'dynamic_heatmap'):
            self.dynamic_heatmap.update_heatmap(fused_prob)

        self.fusion_display.setText(f"{final_emotion.upper()} {final_conf:.0%}")

    def update_video_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = QImage(rgb.data, rgb.shape[1], rgb.shape[0], QImage.Format.Format_RGB888)
        self.video_view.setPixmap(
            QPixmap.fromImage(img).scaled(self.video_view.size(), Qt.AspectRatioMode.KeepAspectRatio))

    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择视频文件", "", "Videos (*.mp4 *.avi *.mkv)")
        if path:
            self.video_thread.start_video(path)
            self.audio_thread.start()

    def stop_all(self):
        self.video_thread.stop()
        self.audio_thread.stop()
        self.video_view.clear()
        self.video_view.setText("系统已停止")
        self.fusion_display.setText("READY")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = ModernApp()
    win.show()
    sys.exit(app.exec())