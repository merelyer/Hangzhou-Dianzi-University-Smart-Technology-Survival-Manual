"""
基于 MNIST CNN 的自动数字识别器
使用 Keras 内置 MNIST 数据集训练卷积神经网络，
然后对分割出的手写数字进行高精度识别。
"""

import cv2
import numpy as np
import os
import csv
import sys

# 屏蔽 TensorFlow 日志
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

INPUT_DIR = r"C:\Users\aa257\Desktop\复习\短学期作业\数据集\segmented_digits_v3"
LABELS_CSV = os.path.join(INPUT_DIR, "labels.csv")

def imread_u(path):
    with open(path, 'rb') as f:
        return cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_GRAYSCALE)


def load_segmented_images():
    """加载所有分割出的数字图像，统一resize到28x28"""
    filenames = []
    images = []
    with open(LABELS_CSV, 'r', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            filenames.append(row['filename'])

    print(f"加载 {len(filenames)} 张数字图片...")
    for i, fname in enumerate(filenames):
        img = imread_u(os.path.join(INPUT_DIR, fname))
        if img is None:
            images.append(None)
            continue

        # 二值化 + 裁剪
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 找到数字区域并居中
        coords = cv2.findNonZero(binary)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            # 添加边距
            pad = 4
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(binary.shape[1], x + w + pad)
            y2 = min(binary.shape[0], y + h + pad)
            digit = binary[y1:y2, x1:x2]

            # 缩放到20x20（MNIST标准预处理）
            scale = 20 / max(digit.shape)
            new_w, new_h = max(1, int(digit.shape[1]*scale)), max(1, int(digit.shape[0]*scale))
            if new_w > 0 and new_h > 0:
                digit = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # 放入28x28画布中央
            canvas = np.zeros((28, 28), dtype=np.uint8)
            off_x = (28 - new_w) // 2
            off_y = (28 - new_h) // 2
            canvas[off_y:off_y+new_h, off_x:off_x+new_w] = digit
        else:
            # 没有内容，直接resize
            canvas = cv2.resize(binary, (28, 28), interpolation=cv2.INTER_AREA)

        # 归一化到 [0, 1]
        images.append(canvas.astype(np.float32) / 255.0)

        if (i+1) % 200 == 0:
            print(f"  已加载 {i+1}/{len(filenames)}")

    return filenames, images


def train_mnist_model():
    """在MNIST上训练一个轻量CNN，返回训练好的模型"""
    print("\n下载MNIST数据集...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # 预处理
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    # 转one-hot
    y_train_cat = keras.utils.to_categorical(y_train, 10)
    y_test_cat = keras.utils.to_categorical(y_test, 10)

    print(f"训练集: {x_train.shape[0]} 张, 测试集: {x_test.shape[0]} 张")

    # 构建CNN模型
    model = keras.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3,3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),

        layers.Conv2D(64, (3,3), activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),

        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax'),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )

    print("\n训练CNN模型（约2-3分钟）...")
    # 数据增强
    datagen = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=8,
        width_shift_range=0.08,
        height_shift_range=0.08,
        zoom_range=0.08,
    )
    datagen.fit(x_train)

    history = model.fit(
        datagen.flow(x_train, y_train_cat, batch_size=128),
        epochs=15,
        validation_data=(x_test, y_test_cat),
        verbose=1,
    )

    test_loss, test_acc = model.evaluate(x_test, y_test_cat, verbose=0)
    print(f"\nMNIST测试准确率: {test_acc:.2%}")

    return model


def predict_digits(model, images, filenames):
    """用训练好的模型预测所有数字"""
    print(f"\n预测 {len(images)} 张数字图片...")

    # 转换为模型输入格式
    x = np.array([img for img in images if img is not None])
    x = np.expand_dims(x, -1)  # (N, 28, 28, 1)

    # 预测
    probs = model.predict(x, batch_size=256, verbose=0)
    predictions = np.argmax(probs, axis=1)
    confidences = np.max(probs, axis=1)

    # 构建结果
    results = {}
    valid_idx = 0
    for i, fname in enumerate(filenames):
        if images[i] is not None:
            results[fname] = (str(predictions[valid_idx]), float(confidences[valid_idx]))
            valid_idx += 1
        else:
            results[fname] = ('?', 0.0)

    return results


def main():
    print("=" * 60)
    print("MNIST-CNN 自动数字识别器")
    print("=" * 60)

    # 1. 训练模型
    model = train_mnist_model()

    # 2. 加载分割图片
    filenames, images = load_segmented_images()

    # 3. 预测
    results = predict_digits(model, images, filenames)

    # 4. 统计
    from collections import Counter
    stats = Counter()
    conf_list = []
    for fname, (digit, conf) in results.items():
        stats[digit] += 1
        conf_list.append(conf)

    print(f"\n{'='*60}")
    print("识别结果统计：")
    print(f"{'='*60}")
    for d in '0123456789?':
        cnt = stats.get(d, 0)
        bar = '█' * (cnt * 40 // max(len(filenames), 1))
        print(f"  数字 {d}: {cnt:4d} 个  {bar}")

    avg_conf = np.mean(conf_list)
    high = sum(1 for c in conf_list if c >= 0.9)
    medium = sum(1 for c in conf_list if 0.5 <= c < 0.9)
    low = sum(1 for c in conf_list if c < 0.5)

    print(f"\n模型置信度：")
    print(f"  高置信度 (≥90%): {high} 张")
    print(f"  中置信度 (50-90%): {medium} 张")
    print(f"  低置信度 (<50%): {low} 张")
    print(f"  平均置信度: {avg_conf:.1%}")

    # 5. 保存
    save_results(results, filenames)
    print(f"\n结果已保存到: {LABELS_CSV}")


def save_results(results, filenames):
    """保存标注结果到CSV"""
    original_rows = {}
    if os.path.exists(LABELS_CSV):
        with open(LABELS_CSV, 'r', encoding='utf-8-sig') as f:
            for row in csv.DictReader(f):
                original_rows[row['filename']] = row

    first = next(iter(original_rows.values())) if original_rows else {}
    fieldnames = list(first.keys()) if first else \
        ['filename','img_source','row','col','x','y','w','h','area','digit_label']

    with open(LABELS_CSV, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(fieldnames)
        for fn in filenames:
            row = []
            for key in fieldnames:
                if key == 'digit_label':
                    row.append(results.get(fn, ('', 0))[0])
                else:
                    row.append(original_rows.get(fn, {}).get(key, ''))
            w.writerow(row)


if __name__ == '__main__':
    main()
