"""
将所有分割出的手写数字图像统一调整为 28×28 像素并标准化像素值。
采用 MNIST 标准预处理方式：保持长宽比缩放到 20×20，居中放入 28×28 画布。
"""

import cv2
import numpy as np
import os
import csv

INPUT_DIR = r"C:\Users\aa257\Desktop\复习\短学期作业\数据集\segmented_digits_v3"
OUTPUT_DIR = r"C:\Users\aa257\Desktop\复习\短学期作业\数据集\digits_28x28"
LABELS_CSV = os.path.join(INPUT_DIR, "labels.csv")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "labels.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 画布大小
CANVAS_SIZE = 28
CONTENT_SIZE = 20  # 数字实际占据的最大区域


def imread_u(path):
    with open(path, 'rb') as f:
        return cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_GRAYSCALE)


def imwrite_u(path, img):
    """保存为 PNG，像素值范围 [0, 255]"""
    # 确保是 uint8 类型
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
    ok, enc = cv2.imencode('.png', img)
    if ok:
        with open(path, 'wb') as f:
            f.write(enc.tobytes())
        return True
    return False


def preprocess_mnist_style(gray):
    """
    按 MNIST 标准预处理：
    1. Otsu 二值化
    2. 找到数字区域并裁剪
    3. 保持长宽比缩放到 20×20
    4. 居中放入 28×28 画布
    5. 归一化到 [0, 1]
    返回: (28x28 float32 归一化图像, 28x28 uint8 图像)
    """
    # 二值化（反转：数字为白色=255）
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 找到数字的实际区域
    coords = cv2.findNonZero(binary)
    if coords is None:
        # 空图像，直接 resize
        resized = cv2.resize(gray, (CANVAS_SIZE, CANVAS_SIZE), interpolation=cv2.INTER_AREA)
        norm = resized.astype(np.float32) / 255.0
        return norm, resized

    x, y, w, h = cv2.boundingRect(coords)

    # 添加少量边距
    pad = 3
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(binary.shape[1], x + w + pad)
    y2 = min(binary.shape[0], y + h + pad)
    digit = binary[y1:y2, x1:x2]

    # 保持长宽比，缩放到 20×20 以内
    scale = CONTENT_SIZE / max(digit.shape[0], digit.shape[1])
    new_h = max(1, int(digit.shape[0] * scale))
    new_w = max(1, int(digit.shape[1] * scale))
    resized = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 居中放入 28×28 画布
    canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8)
    off_y = (CANVAS_SIZE - new_h) // 2
    off_x = (CANVAS_SIZE - new_w) // 2
    canvas[off_y:off_y + new_h, off_x:off_x + new_w] = resized

    # 归一化到 [0, 1]
    normalized = canvas.astype(np.float32) / 255.0

    return normalized, canvas


def main():
    print("=" * 60)
    print("将数字图像统一调整为 28×28 像素并标准化")
    print("=" * 60)

    # 读取文件列表
    filenames = []
    original_rows = {}
    if os.path.exists(LABELS_CSV):
        with open(LABELS_CSV, 'r', encoding='utf-8-sig') as f:
            for row in csv.DictReader(f):
                filenames.append(row['filename'])
                original_rows[row['filename']] = row
    else:
        # 如果没有 labels.csv，直接扫描目录
        for f in sorted(os.listdir(INPUT_DIR)):
            if f.endswith('.png') and f != 'debug':
                filenames.append(f)

    total = len(filenames)
    print(f"共 {total} 张数字图片")
    print(f"输入目录: {INPUT_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")
    print()

    saved_count = 0
    for i, fname in enumerate(filenames):
        img = imread_u(os.path.join(INPUT_DIR, fname))
        if img is None:
            print(f"  [{i+1}/{total}] ⚠ 跳过 {fname}（无法读取）")
            continue

        normalized, uint8_img = preprocess_mnist_style(img)

        # 保存 uint8 版本（便于查看）
        out_path = os.path.join(OUTPUT_DIR, fname)
        imwrite_u(out_path, uint8_img)

        saved_count += 1

        if (i + 1) % 100 == 0 or (i + 1) == total:
            print(f"  [{i+1}/{total}] 已处理 {i+1} 张...")

    # 复制 labels.csv 到输出目录
    if os.path.exists(LABELS_CSV):
        with open(LABELS_CSV, 'r', encoding='utf-8-sig') as f_in:
            with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8-sig') as f_out:
                f_out.write(f_in.read())
        print(f"\n标签文件已复制到: {OUTPUT_CSV}")

    # 保存一份 numpy 数组（标准化后的 float32 数据，方便直接训练）
    print("\n正在生成 numpy 数组文件...")
    images = []
    labels = []
    fname_list = []
    for fname in filenames:
        img = imread_u(os.path.join(INPUT_DIR, fname))
        if img is not None:
            normalized, _ = preprocess_mnist_style(img)
            images.append(normalized)
            # 读取标签（如果有）
            if fname in original_rows:
                lbl = original_rows[fname].get('digit_label', '').strip()
                labels.append(int(lbl) if lbl in '0123456789' else -1)
            else:
                labels.append(-1)
            fname_list.append(fname)

    if images:
        X = np.array(images, dtype=np.float32)  # shape: (N, 28, 28)
        y = np.array(labels, dtype=np.int32)
        info = np.array(fname_list)

        np.savez_compressed(
            os.path.join(OUTPUT_DIR, "digits_28x28.npz"),
            images=X,         # (N, 28, 28) float32, 归一化到 [0, 1]
            labels=y,         # (N,) int32, -1 表示未标注
            filenames=info,   # (N,) 文件名
        )
        labeled_count = np.sum(y >= 0)
        print(f"NumPy 数组已保存: {os.path.join(OUTPUT_DIR, 'digits_28x28.npz')}")
        print(f"  X shape: {X.shape}, dtype: {X.dtype}, 范围: [{X.min():.2f}, {X.max():.2f}]")
        print(f"  y shape: {y.shape}, 已标注: {labeled_count}/{len(y)}")

    print(f"\n{'='*60}")
    print(f"完成！共处理 {saved_count} 张图片")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"  每张图片: 28×28 像素 PNG（uint8, 0-255）")
    print(f"  数组文件: digits_28x28.npz（float32, 归一化到 [0, 1]）")


if __name__ == '__main__':
    main()
