"""
数据增强：将每类数字样本扩充到 500 张（总计 5000 张）。
增强方法：
  1. 旋转（±15°）          — 所有数字
  2. 水平/垂直翻转          — 仅对称数字 (0,1,8)
  3. 亮度变化（±25%）       — 所有数字
  4. 对比度变化（0.7×-1.3×）— 所有数字
  5. 平移+缩放              — 所有数字（替代色调变换，灰度图不适用）
"""

import cv2
import numpy as np
import os
import csv
from collections import Counter

INPUT_DIR = r"C:\Users\aa257\Desktop\复习\短学期作业\数据集\digits_28x28"
OUTPUT_DIR = r"C:\Users\aa257\Desktop\复习\短学期作业\数据集\digits_augmented"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_PER_CLASS = 500
CANVAS_SIZE = 28

# 对称数字 — 翻转后标签不变
SYMMETRIC_DIGITS = {0, 1, 8}      # 水平翻转后不变
VSYMMETRIC_DIGITS = {0, 1, 8}     # 垂直翻转后不变

# 增强参数
ROTATION_RANGE = 15          # 旋转角度范围（度）
BRIGHTNESS_RANGE = 0.25      # 亮度变化范围（比例）
CONTRAST_RANGE = (0.7, 1.3)  # 对比度变化范围
SHIFT_RANGE = 3              # 平移范围（像素）
ZOOM_RANGE = (0.85, 1.15)    # 缩放范围


def imread_u(path):
    with open(path, 'rb') as f:
        return cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_GRAYSCALE)


def imwrite_u(path, img):
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
    ok, enc = cv2.imencode('.png', img)
    if ok:
        with open(path, 'wb') as f:
            f.write(enc.tobytes())
        return True
    return False


# ============================================================
#  增强函数（均在 28×28 uint8 图像上操作）
# ============================================================

def augment_rotation(img, angle=None):
    """旋转图像（±ROTATION_RANGE 度），保持 28×28 尺寸"""
    if angle is None:
        angle = np.random.uniform(-ROTATION_RANGE, ROTATION_RANGE)
    h, w = img.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h),
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return rotated


def augment_flip_horizontal(img):
    """水平翻转"""
    return cv2.flip(img, 1)


def augment_flip_vertical(img):
    """垂直翻转"""
    return cv2.flip(img, 0)


def augment_brightness(img, delta=None):
    """改变亮度：img ± delta*255"""
    if delta is None:
        delta = np.random.uniform(-BRIGHTNESS_RANGE, BRIGHTNESS_RANGE)
    result = img.astype(np.float32) + delta * 255.0
    result = np.clip(result, 0, 255).astype(np.uint8)
    return result


def augment_contrast(img, factor=None):
    """改变对比度：以 128 为中心拉伸"""
    if factor is None:
        factor = np.random.uniform(*CONTRAST_RANGE)
    result = (img.astype(np.float32) - 128) * factor + 128
    result = np.clip(result, 0, 255).astype(np.uint8)
    return result


def augment_shift(img, dx=None, dy=None):
    """平移图像（±SHIFT_RANGE 像素）"""
    if dx is None:
        dx = np.random.randint(-SHIFT_RANGE, SHIFT_RANGE + 1)
    if dy is None:
        dy = np.random.randint(-SHIFT_RANGE, SHIFT_RANGE + 1)
    h, w = img.shape
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    shifted = cv2.warpAffine(img, M, (w, h),
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return shifted


def augment_zoom(img, factor=None):
    """缩放后居中放回 28×28。放大时裁剪中心，缩小时居中补零。"""
    if factor is None:
        factor = np.random.uniform(*ZOOM_RANGE)
    h, w = img.shape
    new_h, new_w = int(h * factor), int(w * factor)
    if new_h < 4 or new_w < 4:
        return img

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    if factor <= 1.0:
        # 缩小 → 居中放入画布
        canvas = np.zeros((h, w), dtype=np.uint8)
        off_y = (h - new_h) // 2
        off_x = (w - new_w) // 2
        canvas[off_y:off_y + new_h, off_x:off_x + new_w] = resized
        return canvas
    else:
        # 放大 → 裁剪中心区域
        crop_y = (new_h - h) // 2
        crop_x = (new_w - w) // 2
        return resized[crop_y:crop_y + h, crop_x:crop_x + w]


# ============================================================
#  组合增强流水线
# ============================================================

def apply_random_augmentations(img, digit_label):
    """
    对给定图像随机应用若干增强，返回增强后的图像。
    根据数字类别决定可用的增强类型。
    """
    result = img.copy()
    methods_used = []

    # 所有数字都随机选择 2-4 种增强组合（每种有概率被选中）
    available_methods = []

    # 1. 旋转（所有数字）
    available_methods.append(('rotate', lambda: augment_rotation(result)))

    # 2. 亮度变化（所有数字）
    available_methods.append(('brightness', lambda: augment_brightness(result)))

    # 3. 对比度变化（所有数字）
    available_methods.append(('contrast', lambda: augment_contrast(result)))

    # 4. 平移（所有数字）
    available_methods.append(('shift', lambda: augment_shift(result)))

    # 5. 缩放（所有数字）
    available_methods.append(('zoom', lambda: augment_zoom(result)))

    # 6. 水平翻转 — 仅对称数字
    if digit_label in SYMMETRIC_DIGITS:
        available_methods.append(('hflip', lambda: augment_flip_horizontal(result)))

    # 7. 垂直翻转 — 仅对称数字
    if digit_label in VSYMMETRIC_DIGITS:
        available_methods.append(('vflip', lambda: augment_flip_vertical(result)))

    # 随机选择 2~4 种方法
    n_methods = np.random.randint(2, min(5, len(available_methods) + 1))
    chosen = np.random.choice(len(available_methods), size=n_methods, replace=False)

    for idx in sorted(chosen):
        name, func = available_methods[idx]
        result = func()
        methods_used.append(name)

    return result


def generate_augmented_samples(img, digit_label, count):
    """为一个原始样本生成 count 个增强版本"""
    samples = []
    for _ in range(count):
        aug = apply_random_augmentations(img, digit_label)
        samples.append(aug)
    return samples


# ============================================================
#  主流程
# ============================================================

def main():
    print("=" * 60)
    print("数据增强 — 每类扩充到 500 张")
    print("=" * 60)

    # 加载原始数据
    data = np.load(os.path.join(INPUT_DIR, "digits_28x28.npz"))
    images = data['images']          # (N, 28, 28) float32 [0,1]
    labels = data['labels']          # (N,) int32
    filenames = data['filenames']    # (N,) str

    # 按类别分组
    classes = {}
    for i in range(len(labels)):
        lbl = int(labels[i])
        if lbl < 0:
            continue
        if lbl not in classes:
            classes[lbl] = []
        classes[lbl].append(i)

    print(f"\n原始数据: {len(labels)} 张")
    for d in sorted(classes.keys()):
        print(f"  数字 {d}: {len(classes[d])} 张 → 需补充 {max(0, TARGET_PER_CLASS - len(classes[d]))} 张")
    print(f"\n目标: 每类 {TARGET_PER_CLASS} 张，共 {TARGET_PER_CLASS * 10} 张")

    # 生成增强数据
    all_images = []
    all_labels = []
    all_sources = []  # 记录来源

    for digit in range(10):
        indices = classes.get(digit, [])
        original_count = len(indices)
        needed = max(0, TARGET_PER_CLASS - original_count)

        # 先放入所有原始样本
        for idx in indices:
            all_images.append(images[idx])
            all_labels.append(digit)
            all_sources.append('original')

        if needed == 0:
            print(f"\n数字 {digit}: {original_count} 张 → 无需补充")
            continue

        # 计算每个原始样本需要生成多少增强版本
        per_sample = needed // original_count
        remainder = needed % original_count

        print(f"\n数字 {digit}: {original_count} → {TARGET_PER_CLASS}（+{needed}）")
        print(f"  每张原始图生成 ~{per_sample}+{remainder} 张增强")

        for j, idx in enumerate(indices):
            # 原始图像转为 uint8 用于 OpenCV 增强
            img_uint8 = (images[idx] * 255).astype(np.uint8)

            # 该样本需生成的增强数量
            to_generate = per_sample + (1 if j < remainder else 0)

            if to_generate > 0:
                aug_samples = generate_augmented_samples(img_uint8, digit, to_generate)
                for aug in aug_samples:
                    # 转回 float32 [0, 1]
                    all_images.append(aug.astype(np.float32) / 255.0)
                    all_labels.append(digit)
                    all_sources.append(f'aug_{filenames[idx]}')

            if (j + 1) % 30 == 0:
                print(f"  已处理 {j+1}/{original_count} 个原始样本...")

    # 保存结果
    X = np.array(all_images, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int32)
    sources = np.array(all_sources)

    npz_path = os.path.join(OUTPUT_DIR, "digits_augmented.npz")
    np.savez_compressed(npz_path, images=X, labels=y, sources=sources)
    print(f"\n{'='*60}")
    print(f"增强完成！")
    print(f"  总样本数: {len(X)}")
    print(f"  X shape: {X.shape}, dtype: {X.dtype}, 范围: [{X.min():.2f}, {X.max():.2f}]")
    print(f"  y shape: {y.shape}")

    # 每类最终统计
    final_dist = Counter(y.tolist())
    print(f"\n  最终各类别分布:")
    for d in sorted(final_dist.keys()):
        orig = len(classes.get(d, []))
        aug = final_dist[d] - orig
        print(f"    数字 {d}: {final_dist[d]} 张（原始 {orig} + 增强 {aug}）")

    print(f"\n  保存到: {npz_path}")

    # 同时保存每类 500 张的 PNG 预览（可选，放子目录）
    preview_dir = os.path.join(OUTPUT_DIR, "preview_per_class")
    os.makedirs(preview_dir, exist_ok=True)
    for digit in range(10):
        digit_indices = [i for i, lbl in enumerate(all_labels) if lbl == digit]
        # 保存该类前 10 张作为预览
        for k, idx in enumerate(digit_indices[:10]):
            img = (X[idx] * 255).astype(np.uint8)
            src_tag = 'orig' if all_sources[idx] == 'original' else 'aug'
            fname = f"digit{digit}_{src_tag}_{k:02d}.png"
            imwrite_u(os.path.join(preview_dir, fname), img)
    print(f"  预览图保存到: {preview_dir}")


if __name__ == '__main__':
    main()
