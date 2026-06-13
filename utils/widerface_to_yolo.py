import os
import cv2


# =========================================================
# 项目根目录
# =========================================================
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

# =========================================================
# 原始数据集目录
# =========================================================
RAW_DIR = os.path.join(
    PROJECT_ROOT,
    "raw_datasets",
    "widerface"
)

# =========================================================
# wider_face_split
# 你的结构：
#
# wider_face_split/
#   └── wider_face_split/
# =========================================================
SPLIT_DIR = os.path.join(
    RAW_DIR,
    "wider_face_split",
    "wider_face_split"
)

# =========================================================
# 图片目录
#
# 你的真实结构：
#
# WIDER_train/
#   └── WIDER_train/
#       └── images/
#
# WIDER_val/
#   └── WIDER_val/
#       └── images/
# =========================================================
TRAIN_IMG_DIR = os.path.join(
    RAW_DIR,
    "WIDER_train",
    "WIDER_train",
    "images"
)

VAL_IMG_DIR = os.path.join(
    RAW_DIR,
    "WIDER_val",
    "WIDER_val",
    "images"
)

# =========================================================
# 标注文件
# =========================================================
TRAIN_TXT = os.path.join(
    SPLIT_DIR,
    "wider_face_train_bbx_gt.txt"
)

VAL_TXT = os.path.join(
    SPLIT_DIR,
    "wider_face_val_bbx_gt.txt"
)

# =========================================================
# 输出目录
# =========================================================
OUT_DIR = os.path.join(
    PROJECT_ROOT,
    "datasets",
    "face"
)

TRAIN_LABEL_DIR = os.path.join(
    OUT_DIR,
    "labels",
    "train"
)

VAL_LABEL_DIR = os.path.join(
    OUT_DIR,
    "labels",
    "val"
)

TRAIN_IMAGE_OUT = os.path.join(
    OUT_DIR,
    "images",
    "train"
)

VAL_IMAGE_OUT = os.path.join(
    OUT_DIR,
    "images",
    "val"
)


# =========================================================
# 创建目录
# =========================================================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


# =========================================================
# 复制图片
# =========================================================
def copy_image(src, dst):

    ensure_dir(os.path.dirname(dst))

    if not os.path.exists(dst):

        with open(src, "rb") as fsrc:
            with open(dst, "wb") as fdst:
                fdst.write(fsrc.read())


# =========================================================
# WIDER FACE -> YOLO
# =========================================================
def convert_split(
        gt_path,
        image_root,
        label_out,
        image_out
):

    ensure_dir(label_out)
    ensure_dir(image_out)

    print(f"\n读取标注文件:")
    print(gt_path)

    with open(gt_path, "r") as f:
        lines = f.readlines()

    i = 0

    total_images = 0
    total_faces = 0
    skipped_images = 0

    while i < len(lines):

        # =================================================
        # 图片相对路径
        # =================================================
        img_rel_path = lines[i].strip()
        i += 1

        if i >= len(lines):
            break

        # =================================================
        # 人脸数量
        # =================================================
        try:
            face_count = int(lines[i].strip())
        except Exception:

            print("\n解析失败")
            print("当前行:", lines[i])
            print("索引:", i)

            break

        i += 1

        # =================================================
        # 图片完整路径
        # =================================================
        img_path = os.path.join(
            image_root,
            img_rel_path
        )

        # 前几张打印调试
        if total_images < 3:
            print("\n测试读取:")
            print(img_path)

        img = cv2.imread(img_path)

        # =================================================
        # 图片读取失败
        # =================================================
        if img is None:

            print("\n读取失败:")
            print(img_path)

            # 跳过对应标注
            i += face_count

            skipped_images += 1
            continue

        h, w = img.shape[:2]

        # =================================================
        # 标签输出路径
        # =================================================
        label_path = os.path.join(
            label_out,
            img_rel_path.replace(".jpg", ".txt")
        )

        ensure_dir(os.path.dirname(label_path))

        # =================================================
        # 图片输出路径
        # =================================================
        image_save_path = os.path.join(
            image_out,
            img_rel_path
        )

        ensure_dir(os.path.dirname(image_save_path))

        valid_faces = 0

        # =================================================
        # 写 YOLO 标签
        # =================================================
        with open(label_path, "w") as lf:

            for _ in range(face_count):

                if i >= len(lines):
                    break

                parts = lines[i].strip().split()
                i += 1

                if len(parts) < 4:
                    continue

                x, y, bw, bh = map(
                    float,
                    parts[:4]
                )

                # =============================================
                # 无效框过滤
                # =============================================
                if bw <= 0 or bh <= 0:
                    continue

                # =============================================
                # 超小脸过滤（推荐）
                # =============================================
                if bw < 8 or bh < 8:
                    continue

                # =============================================
                # YOLO格式
                # =============================================
                xc = (x + bw / 2) / w
                yc = (y + bh / 2) / h

                nw = bw / w
                nh = bh / h

                lf.write(
                    f"0 "
                    f"{xc:.6f} "
                    f"{yc:.6f} "
                    f"{nw:.6f} "
                    f"{nh:.6f}\n"
                )

                valid_faces += 1

        # =================================================
        # 空标签删除
        # =================================================
        if valid_faces == 0:

            if os.path.exists(label_path):
                os.remove(label_path)

            continue

        # =================================================
        # 复制图片
        # =================================================
        copy_image(
            img_path,
            image_save_path
        )

        total_images += 1
        total_faces += valid_faces

        # =================================================
        # 进度
        # =================================================
        if total_images % 1000 == 0:

            print(
                f"\n已处理 {total_images} 张图片 "
                f"| {total_faces} 张人脸"
            )

    # =====================================================
    # 统计
    # =====================================================
    print("\n====================================")
    print("转换完成")
    print("有效图片:", total_images)
    print("有效人脸:", total_faces)
    print("跳过图片:", skipped_images)
    print("====================================")


# =========================================================
# 生成 face.yaml
# =========================================================
def create_yaml():

    ensure_dir(OUT_DIR)

    yaml_path = os.path.join(
        OUT_DIR,
        "face.yaml"
    )

    yaml_content = """path: datasets/face

train: images/train
val: images/val

nc: 1

names:
  0: face
"""

    with open(
            yaml_path,
            "w",
            encoding="utf-8"
    ) as f:

        f.write(yaml_content)

    print("\n已生成:")
    print(yaml_path)


# =========================================================
# 主函数
# =========================================================
def main():

    print("\n===================================")
    print("WIDER FACE -> YOLO 转换")
    print("===================================")

    # =====================================================
    # 检查路径
    # =====================================================
    print("\n检查路径:")

    print("\nTRAIN_TXT:")
    print(TRAIN_TXT)
    print("存在:", os.path.exists(TRAIN_TXT))

    print("\nVAL_TXT:")
    print(VAL_TXT)
    print("存在:", os.path.exists(VAL_TXT))

    print("\nTRAIN_IMG_DIR:")
    print(TRAIN_IMG_DIR)
    print("存在:", os.path.exists(TRAIN_IMG_DIR))

    print("\nVAL_IMG_DIR:")
    print(VAL_IMG_DIR)
    print("存在:", os.path.exists(VAL_IMG_DIR))

    # =====================================================
    # TRAIN
    # =====================================================
    print("\n========== 转换 TRAIN ==========")

    convert_split(
        TRAIN_TXT,
        TRAIN_IMG_DIR,
        TRAIN_LABEL_DIR,
        TRAIN_IMAGE_OUT
    )

    # =====================================================
    # VAL
    # =====================================================
    print("\n========== 转换 VAL ==========")

    convert_split(
        VAL_TXT,
        VAL_IMG_DIR,
        VAL_LABEL_DIR,
        VAL_IMAGE_OUT
    )

    # =====================================================
    # face.yaml
    # =====================================================
    create_yaml()

    print("\n===================================")
    print("全部完成")
    print("===================================")


# =========================================================
# 启动
# =========================================================
if __name__ == "__main__":
    main()