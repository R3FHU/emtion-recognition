import os
import cv2
import torch
import librosa
import numpy as np
import joblib
from moviepy import VideoFileClip
from tqdm import tqdm
from PIL import Image

from ultralytics import YOLO
from transformers import AutoImageProcessor, AutoModelForImageClassification

# 导入你现有的模型结构
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import AudioMLP, CANONICAL_EMOTIONS, MODELS_DIR

# ================== 1. 路径配置 ==================
RAVDESS_VIDEO_DIR = r"E:\python_projects\raw_datasets\RAVDESS_Video"  # 请替换为你的实际视频目录
PROCESSED_DIR = r"E:\python_projects\data\processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

# RAVDESS 标签映射 (文件名第3段，其中02和01都映射为neutral以对齐7分类)
RAVDESS_MAP = {
    '01': 'neutral', '02': 'neutral', '03': 'happy',
    '04': 'sad', '05': 'angry', '06': 'fear',
    '07': 'disgust', '08': 'surprise'
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ================== 2. 加载基模型 ==================
def init_models():
    print("⏳ 正在加载预训练模型...")
    # 1. 人脸模型 YOLO
    yolo_path = os.path.join(MODELS_DIR, "runs/detect/runs/face/yolo11n_maxperf/weights/best.pt")
    face_model = YOLO(yolo_path)

    # 2. 视觉特征提取模型 ViT
    vit_path = os.path.join(MODELS_DIR, "vit-face-expression").replace("\\", "/")
    if not os.path.exists(vit_path): vit_path = "trpakov/vit-face-expression"
    processor = AutoImageProcessor.from_pretrained(vit_path)
    # 注意这里：我们需要输出隐藏层状态
    emotion_model = AutoModelForImageClassification.from_pretrained(vit_path, output_hidden_states=True).to(DEVICE)
    emotion_model.eval()

    # 3. 音频特征提取模型 AudioMLP
    scaler = joblib.load(os.path.join(MODELS_DIR, 'audio_scaler.pkl'))
    le = joblib.load(os.path.join(MODELS_DIR, 'label_encoder.pkl'))
    audio_model = AudioMLP(input_dim=120, num_classes=len(le.classes_)).to(DEVICE)
    audio_model.load_state_dict(torch.load(os.path.join(MODELS_DIR, 'best_audio_model.pt'), map_location=DEVICE))
    audio_model.eval()

    return face_model, processor, emotion_model, audio_model, scaler


# ================== 3. 特征提取函数 ==================
def get_audio_feature(audio_path, model, scaler):
    """提取倒数第二层的 128维 音频特征"""
    y, sr = librosa.load(audio_path, sr=22050)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    feat = np.hstack((np.mean(mfccs.T, axis=0),
                      np.mean(librosa.feature.delta(mfccs).T, axis=0),
                      np.mean(librosa.feature.delta(mfccs, order=2).T, axis=0)))

    feat_scaled = scaler.transform(feat.reshape(1, -1))

    with torch.no_grad():
        x = torch.FloatTensor(feat_scaled).to(DEVICE)
        # 截取你的 AudioMLP 的前 10 层 (正好是最后一个 Linear 之前，输出 128 维)
        feature_128d = model.model[:10](x).cpu().numpy()[0]

    return feature_128d


def get_video_feature(video_path, face_model, processor, emotion_model):
    """抽取多帧人脸，提取 ViT 的 768维 隐层特征并取平均"""
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # 均匀采样 3 帧
    sample_indices = [int(frame_count * 0.25), int(frame_count * 0.5), int(frame_count * 0.75)]

    features = []
    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret: continue

        results = face_model.predict(frame, device=DEVICE, imgsz=320, conf=0.4, verbose=False)
        if results and results[0].boxes:
            box = results[0].boxes[0]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            face = frame[max(0, y1):y2, max(0, x1):x2]

            if face.size != 0:
                pil_img = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
                inputs = processor(images=pil_img, return_tensors="pt").to(DEVICE)
                with torch.no_grad():
                    outputs = emotion_model(**inputs)
                    # 提取最后一层隐藏状态的 CLS token (形状: 1, 768)
                    hidden_state = outputs.hidden_states[-1][:, 0, :].cpu().numpy()[0]
                    features.append(hidden_state)

    cap.release()
    if len(features) == 0:
        return np.zeros(768)  # 没找到人脸时返回全0
    return np.mean(features, axis=0)  # 取三帧特征的平均值


# ================== 4. 主干流水线 ==================
def main():
    face_model, processor, emotion_model, audio_model, scaler = init_models()

    all_features = []
    all_labels = []

    # 临时音频存放路径
    temp_audio = os.path.join(PROCESSED_DIR, "temp.wav")

    print("🔥 开始提取音视频融合特征...")
    # 假设你的视频按照 Actor_01, Actor_02... 存放在文件夹中
    for root, dirs, files in os.walk(RAVDESS_VIDEO_DIR):
        for file in tqdm(files):
            if file.endswith(".mp4"):
                parts = file.split('.')[0].split('-')
                if len(parts) >= 3:
                    emo_code = parts[2]
                    label = RAVDESS_MAP.get(emo_code)
                    if not label: continue

                    video_path = os.path.join(root, file)

                    try:
                        # 1. 拆分音频并提取特征
                        clip = VideoFileClip(video_path)
                        clip.audio.write_audiofile(temp_audio, logger=None)
                        audio_feat = get_audio_feature(temp_audio, audio_model, scaler)
                        clip.close()

                        # 2. 提取视觉特征
                        video_feat = get_video_feature(video_path, face_model, processor, emotion_model)

                        # 3. 拼接中期特征 (128 + 768 = 896维)
                        fused_feat = np.hstack((audio_feat, video_feat))

                        all_features.append(fused_feat)
                        all_labels.append(label)
                    except Exception as e:
                        print(f"处理文件 {file} 时出错: {e}")

    if os.path.exists(temp_audio): os.remove(temp_audio)

    # 保存特征包
    X = np.array(all_features)
    y = np.array(all_labels)
    save_path = os.path.join(PROCESSED_DIR, "mid_fusion_features.npz")
    np.savez(save_path, X=X, y=y)

    print(f"\n✅ 提取完成！共成功处理 {len(X)} 个视频。")
    print(f"融合特征维度: {X.shape[1]} 维 (听觉 128 + 视觉 768)")
    print(f"数据集已保存至: {save_path}")


if __name__ == "__main__":
    main()