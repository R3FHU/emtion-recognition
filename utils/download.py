import os
from transformers import AutoImageProcessor, AutoModelForImageClassification

# 获取当前文件的绝对路径，并指向 models/vit-face-expression 文件夹
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_PATH, "models")
save_path = os.path.join(MODELS_DIR, "vit-face-expression")

os.makedirs(save_path, exist_ok=True)

print("正在从云端下载模型，请稍候...")
# 加载云端模型和处理器
processor = AutoImageProcessor.from_pretrained("trpakov/vit-face-expression")
model = AutoModelForImageClassification.from_pretrained("trpakov/vit-face-expression")

print("下载成功，正在保存至本地目录...")
# 将其保存到本地指定路径
processor.save_pretrained(save_path)
model.save_pretrained(save_path)

print(f"✅ 模型已成功保存至: {save_path}")