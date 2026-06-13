from ultralytics import YOLO
import os
import torch


def main():

    # ================== 路径 ==================
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    data_path = os.path.join(
        BASE_DIR, "..", "datasets", "face", "face.yaml"
    )

    # ================== 模型 ==================
    model = YOLO("yolo11n.pt")

    # ================== CUDA优化 ==================
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("⚠️ 未检测到GPU")

    # ================== 训练 ==================
    model.train(

        # ===== 数据 =====
        data=data_path,

        # ===== 基础训练 =====
        epochs=80,
        imgsz=608,
        batch=10,
        # ===== 性能关键 =====
        device=0,
        workers=4,
        cache="disk",
        amp=True,

        # ===== 优化器 =====
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,

        # ===== 学习策略 =====
        cos_lr=True,
        warmup_epochs=3,
        patience=20,

        # ===== 数据增强（降低CPU负担）=====
        mosaic=0.3,
        mixup=0.0,
        fliplr=0.5,

        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.3,

        degrees=0.0,
        perspective=0.0,

        # ===== 损失 =====
        box=7.5,
        cls=0.5,
        dfl=1.5,

        close_mosaic=10,

        # ===== 输出 =====
        project="runs/face",
        name="yolo11n_maxperf",
        exist_ok=True,

        save=True,
        save_period=20
    )


if __name__ == "__main__":
    main()