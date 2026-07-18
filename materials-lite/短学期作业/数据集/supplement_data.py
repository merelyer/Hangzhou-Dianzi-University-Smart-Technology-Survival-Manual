"""
从朋友数据集补充到每类100张。
将朋友数据转换为与我的数据集一致的格式：
  - 28x28 灰度 PNG
  - Otsu 二值化 + 反转（数字白色255，背景黑色0）
  - MNIST 风格居中
"""

import cv2
import numpy as np
import os
import shutil

MY_BASE = r"C:\Users\aa257\Desktop\复习\短学期作业\数据集\digits_by_class"
FRIEND_BASE = r"C:\Users\aa257\Desktop\复习\短学期作业\dataset\dataset"
TARGET_PER_CLASS = 100


def imread_unicode(path, mode='color'):
    with open(path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8)
    flag = cv2.IMREAD_COLOR if mode == 'color' else cv2.IMREAD_GRAYSCALE
    return cv2.imdecode(data, flag)


def imwrite_unicode(path, img):
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
    ok, enc = cv2.imencode('.png', img)
    if ok:
        with open(path, 'wb') as f:
            f.write(enc.tobytes())
        return True
    return False


def convert_to_my_format(img_bgr):
    """
    将朋友的 RGB 图像转换为我的格式：
    1. 转灰度
    2. Otsu 二值化 + 反转（数字=255, 背景=0）
    3. 裁剪数字区域并居中放入 28x28 画布
    """
    # 1. 转灰度
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 2. Otsu 二值化 + 反转
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 3. 找到数字区域，居中
    coords = cv2.findNonZero(binary)
    if coords is None:
        # 空图，直接返回黑色画布
        return np.zeros((28, 28), dtype=np.uint8)

    x, y, w, h = cv2.boundingRect(coords)

    # 加少量边距
    pad = 3
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(binary.shape[1], x + w + pad)
    y2 = min(binary.shape[0], y + h + pad)
    digit = binary[y1:y2, x1:x2]

    # 保持长宽比缩放到 20x20 以内
    scale = 20.0 / max(digit.shape[0], digit.shape[1], 1)
    new_h = max(1, int(digit.shape[0] * scale))
    new_w = max(1, int(digit.shape[1] * scale))
    resized = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 居中放入 28x28 画布
    canvas = np.zeros((28, 28), dtype=np.uint8)
    off_y = (28 - new_h) // 2
    off_x = (28 - new_w) // 2
    canvas[off_y:off_y + new_h, off_x:off_x + new_w] = resized

    return canvas


def main():
    print("=" * 60)
    print("从朋友数据集补充到每类100张")
    print("=" * 60)

    total_supplemented = 0

    for digit in range(10):
        my_folder = os.path.join(MY_BASE, str(digit))
        my_count = len([f for f in os.listdir(my_folder) if f.endswith('.png')])
        needed = max(0, TARGET_PER_CLASS - my_count)

        if needed == 0:
            print(f"\n数字 {digit}: {my_count} 张 — 已满，无需补充")
            continue

        print(f"\n数字 {digit}: {my_count} → 需补充 {needed} 张")

        # 从朋友数据集读取
        friend_folder = os.path.join(FRIEND_BASE, str(digit))
        friend_files = sorted([
            f for f in os.listdir(friend_folder)
            if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ])

        if len(friend_files) < needed:
            print(f"  [WARN] 朋友只有 {len(friend_files)} 张，不足 {needed} 张！")
            needed = len(friend_files)

        supplemented = 0
        for i in range(needed):
            fpath = os.path.join(friend_folder, friend_files[i])
            img = imread_unicode(fpath, 'color')
            if img is None:
                print(f"  [SKIP] 无法读取: {friend_files[i]}")
                continue

            # 转换为我的格式
            converted = convert_to_my_format(img)

            # 生成新文件名（sup = supplement）
            new_name = f"sup_{supplemented:04d}.png"
            dst_path = os.path.join(my_folder, new_name)
            imwrite_unicode(dst_path, converted)
            supplemented += 1

        total_supplemented += supplemented
        print(f"  → 已补充 {supplemented} 张，当前共 {my_count + supplemented} 张")

    # 验证
    print(f"\n{'='*60}")
    print("补充完成！最终各类别数量：")
    print("=" * 60)
    grand_total = 0
    for digit in range(10):
        n = len([f for f in os.listdir(os.path.join(MY_BASE, str(digit))) if f.endswith('.png')])
        bar = '#' * (n // 2)
        print(f"  数字 {digit}: {n:4d} 张  {bar}")
        grand_total += n
    print(f"  总计:   {grand_total} 张")
    print(f"\n本次补充: {total_supplemented} 张")


if __name__ == '__main__':
    main()
