"""
在原始增强分割图像上训练 CNN — 400 训练 + 100 验证
图像预处理：可变尺寸灰度图 → 28×28 MNIST 标准格式
"""

import os, cv2, json, numpy as np
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ============================================================
# 配置
# ============================================================
INPUT_DIR = r"C:\Users\aa257\Desktop\复习\短学期作业\数据集\原始增强分割图像"
OUTPUT_DIR = r"C:\Users\aa257\Desktop\复习\短学期作业\数据集\original_training"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_PER_CLASS = 40
VAL_PER_CLASS = 10
BATCH_SIZE = 16
EPOCHS = 80
LEARNING_RATE = 0.001
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# ============================================================
# 1. 加载 + 预处理
# ============================================================
def imread_u(path):
    with open(path, 'rb') as f:
        return cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_GRAYSCALE)


def preprocess_to_28x28(gray):
    """可变尺寸灰度图 → 28×28 黑底白字二值图"""
    # Otsu 二值化 + 反转（数字=255, 背景=0）
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 找数字区域
    coords = cv2.findNonZero(binary)
    if coords is None:
        return np.zeros((28, 28), dtype=np.uint8)

    x, y, w, h = cv2.boundingRect(coords)
    pad = 3
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(binary.shape[1], x + w + pad), min(binary.shape[0], y + h + pad)
    digit = binary[y1:y2, x1:x2]

    # 保持长宽比缩放到 20×20
    scale = 20.0 / max(digit.shape[0], digit.shape[1], 1)
    nh, nw = max(1, int(digit.shape[0] * scale)), max(1, int(digit.shape[1] * scale))
    resized = cv2.resize(digit, (nw, nh), interpolation=cv2.INTER_AREA)

    # 居中放入 28×28
    canvas = np.zeros((28, 28), dtype=np.uint8)
    off_y, off_x = (28 - nh) // 2, (28 - nw) // 2
    canvas[off_y:off_y + nh, off_x:off_x + nw] = resized
    return canvas


def load_all_data():
    print("=" * 60)
    print("加载原始增强分割图像并预处理为 28x28")
    print("=" * 60)

    all_images = []
    all_labels = []

    for d in range(10):
        folder = os.path.join(INPUT_DIR, str(d))
        files = [f for f in os.listdir(folder) if f.endswith('.png')]
        for f in files:
            img = imread_u(os.path.join(folder, f))
            if img is not None:
                processed = preprocess_to_28x28(img)
                all_images.append(processed.astype(np.float32) / 255.0)
                all_labels.append(d)

    X = np.array(all_images, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int32)
    print(f"  总样本: {len(X)} 张")
    print(f"  类别分布: {np.bincount(y)}")
    return X, y


def split_data(X, y):
    """分层抽样：每类 40 训练 + 10 验证"""
    class_idx = {d: np.where(y == d)[0] for d in range(10)}

    train_idx, val_idx = [], []
    for d in range(10):
        idx = class_idx[d]
        np.random.shuffle(idx)
        train_idx.extend(idx[:TRAIN_PER_CLASS].tolist())
        val_idx.extend(idx[TRAIN_PER_CLASS:TRAIN_PER_CLASS + VAL_PER_CLASS].tolist())

    X_train = np.expand_dims(X[train_idx], axis=-1)
    y_train = y[train_idx]
    X_val = np.expand_dims(X[val_idx], axis=-1)
    y_val = y[val_idx]

    y_train_cat = keras.utils.to_categorical(y_train, 10)
    y_val_cat = keras.utils.to_categorical(y_val, 10)

    print(f"\n数据划分:")
    print(f"  训练集: {len(X_train)} 张  分布: {np.bincount(y_train)}")
    print(f"  验证集: {len(X_val)} 张  分布: {np.bincount(y_val)}")
    return (X_train, y_train, y_train_cat), (X_val, y_val, y_val_cat)


# ============================================================
# 2. CNN 模型
# ============================================================
def build_cnn():
    model = keras.Sequential(name='OriginalCNN')
    model.add(layers.Conv2D(32, (3,3), padding='same', activation='relu',
                            kernel_regularizer=keras.regularizers.l2(1e-4),
                            input_shape=(28,28,1)))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Dropout(0.25))
    model.add(layers.Conv2D(64, (3,3), padding='same', activation='relu',
                            kernel_regularizer=keras.regularizers.l2(1e-4)))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Dropout(0.3))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu',
                            kernel_regularizer=keras.regularizers.l2(1e-4)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10, activation='softmax'))
    return model


# ============================================================
# 3. 训练
# ============================================================
def train_model(model, X_train, y_train_cat, X_val, y_val_cat):
    model.compile(
        optimizer=keras.optimizers.Adam(LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )

    callbacks = [
        keras.callbacks.CSVLogger(os.path.join(OUTPUT_DIR, 'training_log.csv')),
        keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=20,
                                       restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                           patience=8, min_lr=1e-6, verbose=1),
        keras.callbacks.ModelCheckpoint(os.path.join(OUTPUT_DIR, 'best_model.keras'),
                                         monitor='val_accuracy', save_best_only=True, verbose=1),
    ]

    # 在线增强
    augmenter = keras.Sequential([
        layers.RandomRotation(0.04, fill_mode='constant', fill_value=0),
        layers.RandomTranslation(0.12, 0.12, fill_mode='constant', fill_value=0),
        layers.RandomZoom(0.15, 0.15, fill_mode='constant', fill_value=0),
        layers.RandomContrast(0.25),
    ])

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train_cat))
    train_ds = train_ds.shuffle(512).batch(BATCH_SIZE)
    train_ds = train_ds.map(lambda x, y: (augmenter(x, training=True), y),
                            num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val_cat))
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    print("\n" + "=" * 60)
    print(f"开始训练 — {len(X_train)} 训练 / {len(X_val)} 验证")
    print("=" * 60)

    history = model.fit(train_ds, epochs=EPOCHS, validation_data=val_ds,
                        callbacks=callbacks, verbose=2)
    return history


# ============================================================
# 4. 评估
# ============================================================
def evaluate_model(model, X_val, y_val, history):
    from sklearn.metrics import (classification_report, confusion_matrix,
                                  accuracy_score, precision_recall_fscore_support)

    print("\n" + "=" * 60)
    print("验证集评估")
    print("=" * 60)

    y_probs = model.predict(X_val, batch_size=BATCH_SIZE, verbose=0)
    y_pred = np.argmax(y_probs, axis=1)

    acc = accuracy_score(y_val, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_val, y_pred, average='macro', zero_division=0)

    print(f"\n整体指标:")
    print(f"  准确率: {acc:.4f} ({acc*100:.2f}%)")
    print(f"  宏平均精确率: {prec:.4f}")
    print(f"  宏平均召回率: {rec:.4f}")
    print(f"  宏平均 F1:    {f1:.4f}")

    print(f"\n分类报告:")
    print(classification_report(y_val, y_pred, digits=4,
                                target_names=[f'数字{i}' for i in range(10)], zero_division=0))

    cm = confusion_matrix(y_val, y_pred)
    print("混淆矩阵:")
    print("       " + " ".join(f"预{i:2d}" for i in range(10)))
    for i, row in enumerate(cm):
        print(f"  真{i}: " + " ".join(f"{v:4d}" for v in row))

    per_class = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)
    print(f"\n各类别准确率:")
    for i, p in enumerate(per_class):
        bar = '#' * int(p * 50)
        print(f"  数字 {i}: {p:.4f} ({p*100:5.1f}%) {bar}")

    # 保存报告
    report = {
        'accuracy': float(acc),
        'macro_precision': float(prec),
        'macro_recall': float(rec),
        'macro_f1': float(f1),
        'per_class_accuracy': {str(i): float(a) for i, a in enumerate(per_class)},
        'confusion_matrix': cm.tolist(),
    }
    with open(os.path.join(OUTPUT_DIR, 'validation_report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return acc, cm, per_class


# ============================================================
# 5. 可视化
# ============================================================
def plot_results(history, cm):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    epochs = range(1, len(history.history['loss']) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(epochs, history.history['accuracy'], 'b-', lw=2, label='Train')
    axes[0].plot(epochs, history.history['val_accuracy'], 'r-', lw=2, label='Val')
    axes[0].set_title('Accuracy'); axes[0].set_xlabel('Epoch')
    axes[0].legend(); axes[0].grid(alpha=0.3)
    best = np.argmax(history.history['val_accuracy'])
    axes[0].axvline(best+1, color='g', ls='--', alpha=0.5)

    axes[1].plot(epochs, history.history['loss'], 'b-', lw=2, label='Train')
    axes[1].plot(epochs, history.history['val_loss'], 'r-', lw=2, label='Val')
    axes[1].set_title('Loss'); axes[1].set_xlabel('Epoch')
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_curves.png'), dpi=150)
    plt.close()

    # 混淆矩阵
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap='Blues')
    for i in range(10):
        for j in range(10):
            ax.text(j, i, str(cm[i,j]), ha='center', va='center',
                    color='white' if cm[i,j] > cm.max()/2 else 'black', fontsize=11)
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xlabel('Prediction'); ax.set_ylabel('Ground Truth')
    ax.set_title('Confusion Matrix - Validation Set', fontsize=14)
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), dpi=150)
    plt.close()
    print(f"\n图表已保存: {OUTPUT_DIR}")


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 60)
    print("原始增强分割图像 CNN 训练")
    print(f"训练: {TRAIN_PER_CLASS*10} 张 | 验证: {VAL_PER_CLASS*10} 张")
    print("=" * 60)

    X, y = load_all_data()
    (X_train, y_train, y_train_cat), (X_val, y_val, y_val_cat) = split_data(X, y)

    model = build_cnn()
    model.summary()

    history = train_model(model, X_train, y_train_cat, X_val, y_val_cat)
    val_acc, cm, per_class = evaluate_model(model, X_val, y_val, history)

    try:
        plot_results(history, cm)
    except Exception as e:
        print(f"绘图问题: {e}")

    model.save(os.path.join(OUTPUT_DIR, 'final_model.keras'))

    print(f"\n{'='*60}")
    print(f"训练完成! 验证准确率: {val_acc*100:.2f}%")
    print(f"输出目录: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
