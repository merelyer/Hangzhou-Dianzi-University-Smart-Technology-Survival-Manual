"""
CNN 手写数字识别 — 小样本训练版（400 训练 + 100 验证）
针对小样本优化：简约架构 + 强在线增强 + 强正则化
"""

import numpy as np
import os
import json
import csv
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ============================================================
# 配置
# ============================================================
DATA_PATH = r"C:\Users\aa257\Desktop\复习\短学期作业\数据集\digits_28x28\digits_28x28.npz"
OUTPUT_DIR = r"C:\Users\aa257\Desktop\复习\短学期作业\数据集\cnn_model_v2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_PER_CLASS = 40   # 400 训练
VAL_PER_CLASS = 10     # 100 验证
BATCH_SIZE = 16        # 小 batch 适合小数据集
EPOCHS = 100
LEARNING_RATE = 0.0005
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


def load_and_split():
    print("=" * 60)
    print("加载原始标注数据并按类抽样")
    print("=" * 60)

    data = np.load(DATA_PATH)
    X_all = data['images']
    y_all = data['labels']

    valid = y_all >= 0
    X_all = X_all[valid]
    y_all = y_all[valid]

    print(f"  总样本: {len(X_all)} 张（已标注）")
    print(f"  类别分布: {np.bincount(y_all)}")

    class_indices = {d: [] for d in range(10)}
    for i, lbl in enumerate(y_all):
        class_indices[int(lbl)].append(i)

    train_idx, val_idx = [], []
    for d in range(10):
        indices = class_indices[d]
        np.random.shuffle(indices)
        needed = TRAIN_PER_CLASS + VAL_PER_CLASS
        selected = indices[:needed]
        train_idx.extend(selected[:TRAIN_PER_CLASS])
        val_idx.extend(selected[TRAIN_PER_CLASS:TRAIN_PER_CLASS + VAL_PER_CLASS])

    X_train = np.expand_dims(X_all[train_idx], axis=-1)
    y_train = y_all[train_idx]
    X_val = np.expand_dims(X_all[val_idx], axis=-1)
    y_val = y_all[val_idx]

    y_train_cat = keras.utils.to_categorical(y_train, 10)
    y_val_cat = keras.utils.to_categorical(y_val, 10)

    print(f"\n数据划分:")
    print(f"  训练集: {len(X_train)} 张 分布: {np.bincount(y_train)}")
    print(f"  验证集: {len(X_val)} 张 分布: {np.bincount(y_val)}")

    return (X_train, y_train, y_train_cat), (X_val, y_val, y_val_cat)


def build_small_cnn(input_shape=(28, 28, 1), num_classes=10):
    """
    小样本 CNN — 简约但有效
    - 不用 BN（小样本下统计不稳定）
    - 适中的 Dropout
    - L2 正则化防止过拟合
    """
    model = keras.Sequential(name="SmallCNN")

    # 卷积块 1
    model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu',
                            kernel_regularizer=keras.regularizers.l2(1e-4),
                            input_shape=input_shape, name='Conv1'))
    model.add(layers.MaxPooling2D((2, 2), name='Pool1'))
    model.add(layers.Dropout(0.25, name='Drop1'))

    # 卷积块 2
    model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu',
                            kernel_regularizer=keras.regularizers.l2(1e-4),
                            name='Conv2'))
    model.add(layers.MaxPooling2D((2, 2), name='Pool2'))
    model.add(layers.Dropout(0.3, name='Drop2'))

    # 分类头
    model.add(layers.Flatten(name='Flatten'))
    model.add(layers.Dense(128, activation='relu',
                            kernel_regularizer=keras.regularizers.l2(1e-4),
                            name='FC'))
    model.add(layers.Dropout(0.5, name='Drop3'))
    model.add(layers.Dense(num_classes, activation='softmax', name='Output'))

    return model


def compile_model(model):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


def train_model(model, X_train, y_train_cat, X_val, y_val_cat, X_train_raw, y_train_raw):
    """训练 + 完整监控"""

    # CSV 日志
    csv_logger = keras.callbacks.CSVLogger(
        filename=os.path.join(OUTPUT_DIR, 'training_log.csv'),
        separator=',', append=False,
    )

    # 早停
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=25,
        restore_best_weights=True,
        verbose=1,
    )

    # 学习率衰减
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-6,
        verbose=1,
    )

    # 模型检查点
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(OUTPUT_DIR, 'best_model.keras'),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1,
    )

    # 自定义监控 — 每个 epoch 记录详细信息
    monitor_data = []

    class EpochMonitor(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            record = {
                'epoch': int(epoch) + 1,
                'train_loss': float(logs.get('loss', 0)),
                'train_acc': float(logs.get('accuracy', 0)),
                'val_loss': float(logs.get('val_loss', 0)),
                'val_acc': float(logs.get('val_accuracy', 0)),
                'lr': float(self.model.optimizer.learning_rate.numpy()),
                'timestamp': datetime.now().isoformat(),
            }
            monitor_data.append(record)

    monitor = EpochMonitor()

    callbacks = [csv_logger, early_stop, reduce_lr, checkpoint, monitor]

    print("\n" + "=" * 60)
    print("开始训练...")
    print(f"  训练集: {len(X_train_raw)} 张 | 验证集: {len(X_val)} 张")
    print(f"  Batch={BATCH_SIZE} | LR={LEARNING_RATE} | Max Epochs={EPOCHS}")
    print("=" * 60)

    # === 在线数据增强（训练集）===
    data_augmentation = keras.Sequential([
        layers.RandomRotation(factor=0.04, fill_mode='constant', fill_value=0),
        layers.RandomTranslation(height_factor=0.12, width_factor=0.12, fill_mode='constant', fill_value=0),
        layers.RandomZoom(height_factor=0.15, width_factor=0.15, fill_mode='constant', fill_value=0),
        layers.RandomContrast(factor=0.25),
    ], name='Augmentation')

    # 训练 pipeline
    train_ds = tf.data.Dataset.from_tensor_slices((X_train_raw, y_train_raw))
    train_ds = train_ds.shuffle(512).batch(BATCH_SIZE)
    train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y),
                            num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val_cat))
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # 使用 class_weight 处理类别不均衡
    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=2,
    )

    with open(os.path.join(OUTPUT_DIR, 'monitor.json'), 'w', encoding='utf-8') as f:
        json.dump(monitor_data, f, indent=2, ensure_ascii=False)

    print(f"\n日志已保存: {OUTPUT_DIR}/monitor.json")
    print(f"CSV 已保存:  {OUTPUT_DIR}/training_log.csv")
    return history


def evaluate_on_validation(model, X_val, y_val):
    from sklearn.metrics import (classification_report, confusion_matrix,
                                  accuracy_score, precision_recall_fscore_support)

    print("\n" + "=" * 60)
    print("验证集评估")
    print("=" * 60)

    y_probs = model.predict(X_val, batch_size=BATCH_SIZE, verbose=0)
    y_pred = np.argmax(y_probs, axis=1)

    acc = accuracy_score(y_val, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_val, y_pred, average='macro', zero_division=0
    )

    print(f"\n整体指标:")
    print(f"  准确率: {acc:.4f} ({acc*100:.2f}%)")
    print(f"  宏平均精确率: {precision:.4f}")
    print(f"  宏平均召回率: {recall:.4f}")
    print(f"  宏平均 F1:    {f1:.4f}")

    print(f"\n--- 分类报告 ---")
    print(classification_report(y_val, y_pred, digits=4,
                                target_names=[f'数字{i}' for i in range(10)],
                                zero_division=0))

    cm = confusion_matrix(y_val, y_pred)
    print("--- 混淆矩阵 ---")
    header = "        " + " ".join(f"预{i:2d}" for i in range(10))
    print(header)
    print("       " + "-" * 60)
    for i, row in enumerate(cm):
        print(f"  真{i}:  " + " ".join(f"{v:4d}" for v in row))

    per_class_acc = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)
    print(f"\n--- 各类别准确率 ---")
    for i, pacc in enumerate(per_class_acc):
        bar = '#' * int(pacc * 50)
        print(f"  数字 {i}: {pacc:.4f} ({pacc*100:5.1f}%) {bar}")

    report = {
        'accuracy': float(acc),
        'macro_precision': float(precision),
        'macro_recall': float(recall),
        'macro_f1': float(f1),
        'per_class_accuracy': {str(i): float(a) for i, a in enumerate(per_class_acc)},
        'confusion_matrix': cm.tolist(),
        'n_val_samples': int(len(y_val)),
    }
    with open(os.path.join(OUTPUT_DIR, 'validation_report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n报告已保存: {OUTPUT_DIR}/validation_report.json")
    return acc, cm, per_class_acc


def plot_training(history):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    epochs = range(1, len(history.history['loss']) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss
    axes[0].plot(epochs, history.history['loss'], 'b-', lw=2, label='Train')
    axes[0].plot(epochs, history.history['val_loss'], 'r-', lw=2, label='Val')
    axes[0].set_title('Loss', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Epoch'); axes[0].legend(); axes[0].grid(alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, history.history['accuracy'], 'b-', lw=2, label='Train')
    axes[1].plot(epochs, history.history['val_accuracy'], 'r-', lw=2, label='Val')
    axes[1].set_title('Accuracy', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Epoch'); axes[1].legend(); axes[1].grid(alpha=0.3)
    best = np.argmax(history.history['val_accuracy'])
    axes[1].axvline(x=best + 1, color='g', linestyle='--', alpha=0.5, label=f'Best epoch {best+1}')
    axes[1].legend()

    # LR
    lr_hist = history.history.get('learning_rate', [LEARNING_RATE] * len(epochs))
    axes[2].plot(epochs, lr_hist, 'g-', lw=2)
    axes[2].set_title('Learning Rate', fontsize=13, fontweight='bold')
    axes[2].set_xlabel('Epoch'); axes[2].set_yscale('log'); axes[2].grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'training_curves.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"训练曲线已保存: {path}")

    # 混淆矩阵
    return path


def plot_confusion_matrix(cm):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap='Blues')
    for i in range(10):
        for j in range(10):
            color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', color=color,
                    fontsize=11, fontweight='bold')
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xlabel('Prediction'); ax.set_ylabel('Ground Truth')
    ax.set_title('Confusion Matrix - Validation Set', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'confusion_matrix.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"混淆矩阵已保存: {path}")


def main():
    print("=" * 60)
    print("CNN 手写数字识别 — 小样本训练")
    print(f"训练: {TRAIN_PER_CLASS * 10} 张 | 验证: {VAL_PER_CLASS * 10} 张")
    print("=" * 60)

    (X_train, y_train, y_train_cat), (X_val, y_val, y_val_cat) = load_and_split()

    print("\n构建模型...")
    model = build_small_cnn()
    model.summary()
    compile_model(model)

    print(f"\n训练配置:")
    print(f"  架构: 2xConv(32,64) + FC(128) + L2 + Dropout")
    print(f"  优化器: Adam(lr={LEARNING_RATE})")
    print(f"  在线增强: 旋转±15°+平移±12%+缩放±15%+对比度±25%")

    history = train_model(model, X_train, y_train_cat, X_val, y_val_cat,
                          X_train, y_train_cat)

    val_acc, cm, per_class = evaluate_on_validation(model, X_val, y_val)

    try:
        plot_training(history)
        plot_confusion_matrix(cm)
    except Exception as e:
        print(f"绘图问题: {e}")

    model.save(os.path.join(OUTPUT_DIR, 'final_model.keras'))

    print(f"\n{'='*60}")
    print("训练完成!")
    print(f"  验证准确率: {val_acc*100:.2f}%")
    print(f"  输出目录:    {OUTPUT_DIR}")
    print(f"  最佳模型:    best_model.keras")
    print(f"  训练日志:    training_log.csv")
    print(f"  训练曲线:    training_curves.png")
    print(f"  混淆矩阵:    confusion_matrix.png")
    print(f"  验证报告:    validation_report.json")


if __name__ == '__main__':
    main()
