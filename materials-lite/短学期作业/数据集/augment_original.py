"""
对原始分割风格图像做数据增强 — 每类扩充到 500 张
保持原始风格：变尺寸、灰色纸底、墨迹数字
增强方法：翻转、旋转、亮度、对比度、色调模拟
"""

import os, cv2, numpy as np

INPUT_DIR = r"C:\Users\aa257\Desktop\复习\短学期作业\数据集\original_segmented_by_class"
OUTPUT_DIR = r"C:\Users\aa257\Desktop\复习\短学期作业\数据集\original_augmented_by_class"

TARGET = 500
SAFE_FLIP_DIGITS = {0, 1, 8}  # 翻转后标签不变的对称数字

for d in range(10):
    os.makedirs(os.path.join(OUTPUT_DIR, str(d)), exist_ok=True)


def imread_u(path):
    with open(path, 'rb') as f:
        return cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_GRAYSCALE)


def imwrite_u(path, img):
    ok, enc = cv2.imencode('.png', img)
    if ok:
        with open(path, 'wb') as f:
            f.write(enc.tobytes())


def augment_image(img, digit_label, rng):
    """对一张原始分割图随机组合多种增强"""
    result = img.copy()
    methods = []

    # ---- 1. 旋转 (-10 ~ +10 度) ----
    if rng.random() < 0.6:
        angle = rng.uniform(-10, 10)
        h, w = result.shape
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        result = cv2.warpAffine(result, M, (w, h),
                                borderMode=cv2.BORDER_REPLICATE)
        methods.append('rotate_%.0f' % angle)

    # ---- 2. 翻转（仅对称数字 0,1,8）----
    if digit_label in SAFE_FLIP_DIGITS:
        if rng.random() < 0.2:
            result = cv2.flip(result, 1)  # 水平
            methods.append('hflip')
        if rng.random() < 0.2:
            result = cv2.flip(result, 0)  # 垂直
            methods.append('vflip')

    # ---- 3. 亮度变化 (-20 ~ +20) ----
    if rng.random() < 0.7:
        delta = rng.uniform(-20, 20)
        result = result.astype(np.float32) + delta
        result = np.clip(result, 0, 255).astype(np.uint8)
        methods.append('bright_%.0f' % delta)

    # ---- 4. 对比度变化 (0.85 ~ 1.15) ----
    if rng.random() < 0.7:
        factor = rng.uniform(0.85, 1.15)
        mean_val = result.mean()
        result = ((result.astype(np.float32) - mean_val) * factor + mean_val)
        result = np.clip(result, 0, 255).astype(np.uint8)
        methods.append('contrast_%.2f' % factor)

    # ---- 5. 色调模拟（灰度→伪彩色调偏移→灰度）----
    if rng.random() < 0.4:
        # 转 BGR 后在 HSV 空间偏移色调
        bgr = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        hue_shift = rng.uniform(-20, 20)
        hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * rng.uniform(0.9, 1.1), 0, 255)
        hsv = hsv.astype(np.uint8)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        result = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        methods.append('hue_%.0f' % hue_shift)

    # ---- 6. 轻微高斯噪声（纸张纹理）----
    if rng.random() < 0.5:
        noise = rng.normal(0, 2.5, result.shape).astype(np.float32)
        result = result.astype(np.float32) + noise
        result = np.clip(result, 0, 255).astype(np.uint8)
        methods.append('noise')

    # ---- 7. 轻微缩放 (0.93 ~ 1.07) ----
    if rng.random() < 0.5:
        scale = rng.uniform(0.93, 1.07)
        h, w = result.shape
        new_w, new_h = max(8, int(w * scale)), max(8, int(h * scale))
        resized = cv2.resize(result, (new_w, new_h), interpolation=cv2.INTER_AREA)
        if scale < 1.0:
            # 缩小 → 居中放入原尺寸画布
            canvas = np.full((h, w), result[0, 0].item(), dtype=np.uint8)
            off_y, off_x = (h - new_h) // 2, (w - new_w) // 2
            canvas[off_y:off_y + new_h, off_x:off_x + new_w] = resized
            result = canvas
        else:
            # 放大 → 裁剪中心
            crop_y, crop_x = (new_h - h) // 2, (new_w - w) // 2
            result = resized[crop_y:crop_y + h, crop_x:crop_x + w]
        methods.append('scale_%.2f' % scale)

    return result


def main():
    rng = np.random.RandomState(42)
    total_gen = 0

    for digit in range(10):
        src_folder = os.path.join(INPUT_DIR, str(digit))
        dst_folder = os.path.join(OUTPUT_DIR, str(digit))

        # 加载该类所有种子图像
        seed_files = sorted([f for f in os.listdir(src_folder) if f.endswith('.png')])
        seeds = []
        for f in seed_files:
            img = imread_u(os.path.join(src_folder, f))
            if img is not None and img.shape[0] > 5 and img.shape[1] > 5:
                seeds.append(img)

        current = len(seeds)
        needed = max(0, TARGET - current)

        # 先复制所有原始图像
        for f in seed_files:
            img = imread_u(os.path.join(src_folder, f))
            if img is not None:
                imwrite_u(os.path.join(dst_folder, f), img)

        # 生成增强图像
        generated = 0
        for i in range(needed):
            seed = seeds[rng.randint(0, len(seeds))]
            aug = augment_image(seed, digit, rng)
            fname = 'aug_%05d.png' % i
            imwrite_u(os.path.join(dst_folder, fname), aug)
            generated += 1

        final = len([f for f in os.listdir(dst_folder) if f.endswith('.png')])
        print('digit %d: %d orig + %d aug = %d total' % (digit, current, generated, final))
        total_gen += generated

    # 汇总
    print('\n' + '=' * 50)
    grand = 0
    for d in range(10):
        n = len([f for f in os.listdir(os.path.join(OUTPUT_DIR, str(d))) if f.endswith('.png')])
        grand += n
        print('digit %d: %d' % (d, n))
    print('total: %d' % grand)
    print('output: %s' % OUTPUT_DIR)


if __name__ == '__main__':
    main()
