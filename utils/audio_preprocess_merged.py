import os
import librosa
import numpy as np
from tqdm import tqdm

# ==========================================
# 1. 路径与全局配置
# ==========================================
RAVDESS_DIR = r"E:\python_projects\raw_datasets\RAVDESS"
TESS_DIR = r"E:\python_projects\raw_datasets\TESS Toronto emotional speech set data"
PROCESSED_DIR = r"E:\python_projects\data\processed"

CANONICAL_EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# RAVDESS 情绪编码映射表 (第3个数字)
# 注意：RAVDESS 中的 02 是 Calm (平静)，我们将其合并到 neutral 中以保持 7 分类
RAVDESS_MAP = {
    '01': 'neutral', '02': 'neutral', '03': 'happy',
    '04': 'sad', '05': 'angry', '06': 'fear',
    '07': 'disgust', '08': 'surprise'
}


# ==========================================
# 2. 标签解析函数
# ==========================================
def parse_tess_label(folder_name):
    """从 TESS 的文件夹名称解析出标准标签"""
    name_lower = folder_name.lower()
    if 'angry' in name_lower: return 'angry'
    if 'disgust' in name_lower: return 'disgust'
    if 'fear' in name_lower: return 'fear'
    if 'happy' in name_lower: return 'happy'
    if 'neutral' in name_lower: return 'neutral'
    if 'sad' in name_lower: return 'sad'
    if 'surprise' in name_lower: return 'surprise'
    return None


def parse_ravdess_label(file_name):
    """从 RAVDESS 的文件名称解析出标准标签"""
    # RAVDESS 文件名格式: 03-01-06-01-02-01-12.wav
    parts = file_name.split('-')
    if len(parts) >= 3:
        emo_code = parts[2]
        return RAVDESS_MAP.get(emo_code, None)
    return None


# ==========================================
# 3. 特征提取与数据增强 (6变体)
# ==========================================
def extract_features(y, sr):
    """提取 120 维的声学特征 (MFCC + Delta + Delta2)"""
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfccs.T, axis=0)
    mfcc_delta = np.mean(librosa.feature.delta(mfccs).T, axis=0)
    mfcc_delta2 = np.mean(librosa.feature.delta(mfccs, order=2).T, axis=0)
    return np.hstack((mfcc_mean, mfcc_delta, mfcc_delta2))


def augment_and_extract(file_path):
    """进行音频数据增强，并返回 6 组特征列表"""
    try:
        y, sr = librosa.load(file_path, sr=22050)
    except Exception:
        return []

    features_list = []

    # 1. 原始音频
    features_list.append(extract_features(y, sr))
    # 2. 变速 (0.8x)
    features_list.append(extract_features(librosa.effects.time_stretch(y, rate=0.8), sr))
    # 3. 变速 (1.2x)
    features_list.append(extract_features(librosa.effects.time_stretch(y, rate=1.2), sr))
    # 4. 移频 (升调 2 个半音)
    features_list.append(extract_features(librosa.effects.pitch_shift(y, sr=sr, n_steps=2), sr))
    # 5. 移频 (降调 2 个半音)
    features_list.append(extract_features(librosa.effects.pitch_shift(y, sr=sr, n_steps=-2), sr))
    # 6. 加性噪声 (白噪声)
    noise = np.random.randn(len(y))
    y_noise = y + 0.005 * noise
    features_list.append(extract_features(y_noise, sr))

    return features_list


# ==========================================
# 4. 主干流程：融合收集与处理
# ==========================================
def process_merged_datasets():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # 第一步：收集所有文件的绝对路径和对应的标准标签
    file_registry = []

    print(">>> 正在扫描并映射 TESS 数据集...")
    if os.path.exists(TESS_DIR):
        for folder in os.listdir(TESS_DIR):
            folder_path = os.path.join(TESS_DIR, folder)
            if not os.path.isdir(folder_path): continue
            label = parse_tess_label(folder)
            if not label: continue

            for file in os.listdir(folder_path):
                if file.endswith('.wav'):
                    file_registry.append((os.path.join(folder_path, file), label))

    print(">>> 正在扫描并映射 RAVDESS 数据集...")
    if os.path.exists(RAVDESS_DIR):
        for actor_folder in os.listdir(RAVDESS_DIR):
            actor_path = os.path.join(RAVDESS_DIR, actor_folder)
            if not os.path.isdir(actor_path): continue

            for file in os.listdir(actor_path):
                if file.endswith('.wav'):
                    label = parse_ravdess_label(file)
                    if label:
                        file_registry.append((os.path.join(actor_path, file), label))

    print(f"✅ 扫描完毕！共找到 {len(file_registry)} 个有效音频文件。")
    print(">>> 开始进行数据增强与 MFCC 提取 (预计耗时较长，请耐心等待)...")

    all_features = []
    all_labels = []

    # 第二步：遍历注册表，执行增强与特征提取
    for file_path, label in tqdm(file_registry, desc="Processing Audio Pipeline"):
        augmented_feats = augment_and_extract(file_path)
        for feat in augmented_feats:
            all_features.append(feat)
            all_labels.append(label)

    # 第三步：打包保存为 npz
    save_path = os.path.join(PROCESSED_DIR, 'audio_features.npz')
    np.savez(save_path, X=np.array(all_features), y=np.array(all_labels))
    print(f"🎉 融合处理完成！共生成 {len(all_labels)} 个训练样本。")
    print(f"特征文件已保存至: {save_path}")


if __name__ == "__main__":
    process_merged_datasets()