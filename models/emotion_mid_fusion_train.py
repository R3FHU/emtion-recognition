import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report
import joblib

# ================== 1. 路径与配置 ==================
FEATURE_PATH = r"E:\python_projects\data\processed\mid_fusion_features.npz"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'best_mid_fusion_model.pt')
SCALER_SAVE_PATH = os.path.join(BASE_DIR, 'mid_fusion_scaler.pkl')
ENCODER_SAVE_PATH = os.path.join(BASE_DIR, 'mid_fusion_encoder.pkl')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================== 2. 加载与预处理数据 ==================
print(f"⏳ 正在加载中期融合特征: {FEATURE_PATH}")
data = np.load(FEATURE_PATH)
X = data['X']  # 形状应为 (N, 896)
y = data['y']  # 字符串标签

print(f"✅ 数据加载完成！样本数: {X.shape[0]}, 特征维度: {X.shape[1]}")

# 标签编码
le = LabelEncoder()
y_encoded = le.fit_transform(y)
joblib.dump(le, ENCODER_SAVE_PATH)

# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# 标准化特征
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
joblib.dump(scaler, SCALER_SAVE_PATH)

# 转为 PyTorch Tensors
train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)


# ================== 3. 构建跨模态融合网络 ==================
class CrossModalFusionNet(nn.Module):
    def __init__(self, input_dim=896, num_classes=7):
        super(CrossModalFusionNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(256, 64),
            nn.ReLU(),

            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.network(x)


model = CrossModalFusionNet(input_dim=X.shape[1], num_classes=len(le.classes_)).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# ================== 4. 训练与早停逻辑 ==================
epochs = 150
best_val_loss = float('inf')
patience = 20
counter = 0

print(f"\n🔥 开始训练中期融合网络 (设备: {DEVICE})...")
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
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()

    avg_val_loss = val_loss / len(val_loader)
    val_acc = correct / len(X_val)

    # 更新学习率
    scheduler.step(avg_val_loss)

    if (epoch + 1) % 5 == 0:
        print(
            f"Epoch [{epoch + 1:03d}/{epochs}] | Train Loss: {train_loss / len(train_loader):.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")

    # 早停与模型保存
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print(f"🛑 触发早停：验证集 Loss 连续 {patience} 轮未下降。")
            break

# ================== 5. 评估最终融合效果 ==================
print("\n✨ 加载最佳权重生成测试报告...")
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

print(f"\n✅ 融合网络及其预处理器已保存至 {BASE_DIR} 目录！")
print("\n======= 跨模态融合最终评估报告 =======")
print(classification_report(all_labels, all_preds, target_names=le.classes_))