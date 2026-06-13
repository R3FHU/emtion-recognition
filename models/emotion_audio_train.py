import os
import glob
import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report

# ================== 1. 路径与配置 ==================
# 1. 指向你刚才刚刚生成 audio_features.npz 的绝对路径文件夹
FEATURE_DIR = r"E:\python_projects\data\processed"

# 2. 创建 models 文件夹（如果不存在），确保 main.py 能找到
os.makedirs("models", exist_ok=True)

# 3. 将模型和字典统一保存到 models 文件夹下
MODEL_SAVE_PATH = 'best_audio_model.pt'
SCALER_SAVE_PATH = 'audio_scaler.pkl'
ENCODER_SAVE_PATH = 'label_encoder.pkl'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================== 2. 加载与合并数据 ==================
print(f"⏳ 正在读取特征批次文件...")
batch_files = glob.glob(os.path.join(FEATURE_DIR, "*.npz"))
if not batch_files:
    print(f"❌ 错误：在 {FEATURE_DIR} 中没找到 npz 文件！请检查路径或确认预处理脚本已跑完。")
    exit()

X_list, y_list = [], []
for f in batch_files:
    data = np.load(f)
    X_list.append(data['X'])
    y_list.append(data['y'])

X = np.vstack(X_list)
y = np.concatenate(y_list)
print(f"✅ 加载完成！总样本数: {X.shape[0]}, 特征维度: {X.shape[1]}")

# ================== 3. 数据预处理 ==================
# 标签转整数 (0-6)
le = LabelEncoder()
y_encoded = le.fit_transform(y)
joblib.dump(le, ENCODER_SAVE_PATH)

# 划分训练/验证集 (8:2)
X_train, X_val, y_train, y_val = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# 标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
joblib.dump(scaler, SCALER_SAVE_PATH)

# 转换为 PyTorch 张量
train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

# ================== 4. 定义 Robust MLP 网络 ==================
class AudioMLP(nn.Module):
    def __init__(self, input_dim=120, num_classes=7):
        super(AudioMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# 实例化模型
# 这里会自动读取特征的120维和标签的7分类
model = AudioMLP(input_dim=X.shape[1], num_classes=len(le.classes_)).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# ================== 5. 训练循环 + 早停逻辑 ==================
epochs = 100
best_val_loss = float('inf')
patience = 12  # 增加容忍度
counter = 0

print(f"\n🔥 开始训练 (使用设备: {DEVICE})...")
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # 验证
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            val_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()

    val_acc = correct / len(X_val)
    avg_val_loss = val_loss / len(val_loader)

    # 打印进度 (每 2 轮打印一次)
    if (epoch + 1) % 2 == 0:
        print(f"Epoch [{epoch + 1:02d}/{epochs}] | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")

    # 早停检查与保存
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        # 保存为 .pt 格式到 models 文件夹
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print(f"🛑 停止：验证集 Loss 连续 {patience} 轮未下降。")
            break

# ================== 6. 最终评估 ==================
print("\n✨ 加载最佳权重并生成最终报告...")
# 确保这里加载的是新保存到 models 文件夹下的 .pt 文件
model.load_state_dict(torch.load(MODEL_SAVE_PATH))
model.eval()

all_preds = []
all_labels = []
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs = inputs.to(DEVICE)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

print(f"\n✅ 模型及预处理器已成功保存至 models/ 目录下！")
print(classification_report(all_labels, all_preds, target_names=le.classes_))