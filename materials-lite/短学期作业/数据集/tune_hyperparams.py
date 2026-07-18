"""
CNN 超参数调优 — 系统搜索最佳配置
Phase 1: 学习率搜索
Phase 2: 批量大小搜索
Phase 3: 架构搜索（更深 + BN + Dropout）

所有实验使用相同的数据划分（400 训练 / 100 验证）确保公平对比。
"""

import numpy as np
import os
import json
import csv
from datetime import datetime
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 减少日志输出
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ============================================================
# 全局配置
# ============================================================
DATA_PATH = r"C:\Users\aa257\Desktop\复习\短学期作业\数据集\digits_28x28\digits_28x28.npz"
OUTPUT_DIR = r"C:\Users\aa257\Desktop\复习\短学期作业\数据集\cnn_tuning"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_PER_CLASS = 40
VAL_PER_CLASS = 10
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# ============================================================
# 数据加载（所有实验共享）
# ============================================================
def load_fixed_split():
    """加载并固定训练/验证划分（所有实验使用同一划分）"""
    data = np.load(DATA_PATH)
    X_all = data['images']
    y_all = data['labels']
    valid = y_all >= 0
    X_all = X_all[valid]
    y_all = y_all[valid]

    class_indices = {d: [] for d in range(10)}
    for i, lbl in enumerate(y_all):
        class_indices[int(lbl)].append(i)

    train_idx, val_idx = [], []
    for d in range(10):
        indices = class_indices[d]
        np.random.shuffle(indices)
        selected = indices[:TRAIN_PER_CLASS + VAL_PER_CLASS]
        train_idx.extend(selected[:TRAIN_PER_CLASS])
        val_idx.extend(selected[TRAIN_PER_CLASS:])

    X_train = np.expand_dims(X_all[train_idx], axis=-1)
    y_train = y_all[train_idx]
    X_val = np.expand_dims(X_all[val_idx], axis=-1)
    y_val = y_all[val_idx]

    y_train_cat = keras.utils.to_categorical(y_train, 10)
    y_val_cat = keras.utils.to_categorical(y_val, 10)

    return X_train, y_train_cat, X_val, y_val_cat, y_val


# ============================================================
# 模型构建器
# ============================================================

def build_baseline_2conv(dropout1=0.25, dropout2=0.3, dropout_fc=0.5, l2=1e-4):
    """Baseline: 2 Conv + FC"""
    model = keras.Sequential(name='2Conv_Baseline')
    model.add(layers.Conv2D(32, (3,3), padding='same', activation='relu',
                            kernel_regularizer=keras.regularizers.l2(l2),
                            input_shape=(28,28,1)))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Dropout(dropout1))
    model.add(layers.Conv2D(64, (3,3), padding='same', activation='relu',
                            kernel_regularizer=keras.regularizers.l2(l2)))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Dropout(dropout2))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu',
                            kernel_regularizer=keras.regularizers.l2(l2)))
    model.add(layers.Dropout(dropout_fc))
    model.add(layers.Dense(10, activation='softmax'))
    return model


def build_deeper_3conv(dropout1=0.2, dropout2=0.25, dropout3=0.3, dropout_fc=0.5, l2=1e-4):
    """3 Conv + BN + FC — 更深架构"""
    model = keras.Sequential(name='3Conv_BN')
    # Block 1
    model.add(layers.Conv2D(32, (3,3), padding='same', use_bias=False, input_shape=(28,28,1)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(32, (3,3), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Dropout(dropout1))
    # Block 2
    model.add(layers.Conv2D(64, (3,3), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(64, (3,3), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Dropout(dropout2))
    # Block 3
    model.add(layers.Conv2D(128, (3,3), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Dropout(dropout3))
    # FC
    model.add(layers.Flatten())
    model.add(layers.Dense(128, use_bias=False, kernel_regularizer=keras.regularizers.l2(l2)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(dropout_fc))
    model.add(layers.Dense(10, activation='softmax'))
    return model


def build_deep_4conv(dropout1=0.15, dropout2=0.2, dropout3=0.25, dropout4=0.3, dropout_fc=0.5, l2=1e-4):
    """4 Conv + BN + FC — 最深层架构"""
    model = keras.Sequential(name='4Conv_BN')
    def conv_block(filters, dropout_rate):
        model.add(layers.Conv2D(filters, (3,3), padding='same', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.Conv2D(filters, (3,3), padding='same', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling2D((2,2)))
        model.add(layers.Dropout(dropout_rate))

    conv_block(32, dropout1)
    conv_block(64, dropout2)
    conv_block(128, dropout3)
    # Block 4 (no pooling — already 3x3)
    model.add(layers.Conv2D(128, (3,3), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(dropout4))

    model.add(layers.Flatten())
    model.add(layers.Dense(128, use_bias=False, kernel_regularizer=keras.regularizers.l2(l2)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(dropout_fc))
    model.add(layers.Dense(10, activation='softmax'))
    return model


def build_medium_2conv_bn(dropout1=0.25, dropout2=0.3, dropout_fc=0.5, l2=1e-4):
    """2 Conv + BN + FC — 中等架构"""
    model = keras.Sequential(name='2Conv_BN')
    model.add(layers.Conv2D(32, (3,3), padding='same', use_bias=False, input_shape=(28,28,1)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Dropout(dropout1))

    model.add(layers.Conv2D(64, (3,3), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Dropout(dropout2))

    model.add(layers.Flatten())
    model.add(layers.Dense(128, use_bias=False, kernel_regularizer=keras.regularizers.l2(l2)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(dropout_fc))
    model.add(layers.Dense(10, activation='softmax'))
    return model


# ============================================================
# 训练 + 评估
# ============================================================
def run_experiment(name, build_fn, lr, batch_size, X_train, y_train_cat, X_val, y_val_cat, y_val, epochs=80):
    """运行单个实验并返回结果"""
    print(f"\n{'='*60}")
    print(f"[EXP] {name}")
    print(f"  LR={lr}, Batch={batch_size}")
    print(f"{'='*60}")

    tf.random.set_seed(RANDOM_SEED)
    model = build_fn()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )

    # Callbacks
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=20, restore_best_weights=True, verbose=0)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6, verbose=0)

    # 数据增强 pipeline
    augmenter = keras.Sequential([
        layers.RandomRotation(0.04, fill_mode='constant', fill_value=0),
        layers.RandomTranslation(0.12, 0.12, fill_mode='constant', fill_value=0),
        layers.RandomZoom(0.15, 0.15, fill_mode='constant', fill_value=0),
        layers.RandomContrast(0.25),
    ])

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train_cat))
    train_ds = train_ds.shuffle(512).batch(batch_size)
    train_ds = train_ds.map(lambda x, y: (augmenter(x, training=True), y),
                            num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val_cat))
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    t0 = time.time()
    history = model.fit(train_ds, epochs=epochs, validation_data=val_ds,
                        callbacks=[early_stop, reduce_lr], verbose=0)
    train_time = time.time() - t0

    # 评估
    y_probs = model.predict(val_ds, verbose=0)
    y_pred = np.argmax(y_probs, axis=1)

    from sklearn.metrics import accuracy_score
    val_acc = accuracy_score(y_val, y_pred)

    result = {
        'name': name,
        'lr': lr,
        'batch_size': batch_size,
        'val_accuracy': float(val_acc),
        'best_val_acc': float(max(history.history['val_accuracy'])),
        'final_train_acc': float(history.history['accuracy'][-1]),
        'best_train_acc': float(max(history.history['accuracy'])),
        'final_train_loss': float(history.history['loss'][-1]),
        'final_val_loss': float(history.history['val_loss'][-1]),
        'epochs_trained': len(history.history['loss']),
        'train_time_sec': float(train_time),
        'params': int(model.count_params()),
    }

    print(f"  Val Acc: {result['val_accuracy']:.4f} | Best Val: {result['best_val_acc']:.4f}")
    print(f"  Epochs: {result['epochs_trained']} | Time: {train_time:.1f}s | Params: {result['params']:,}")

    return result, model, history


# ============================================================
# 结果汇总与选择最佳
# ============================================================
def print_summary(results):
    print("\n\n" + "=" * 80)
    print("超参数调优结果汇总")
    print("=" * 80)
    print(f"\n{'实验':<35} {'LR':>8} {'Batch':>6} {'ValAcc':>8} {'BestVal':>8} {'Epochs':>7} {'Time':>7} {'Params':>10}")
    print("-" * 100)

    sorted_results = sorted(results, key=lambda r: r['val_accuracy'], reverse=True)
    for i, r in enumerate(sorted_results):
        marker = '>>>' if i == 0 else '   '
        print(f"{marker} {r['name']:<32} {r['lr']:>8.5f} {r['batch_size']:>6d} "
              f"{r['val_accuracy']:>8.4f} {r['best_val_acc']:>8.4f} {r['epochs_trained']:>7d} "
              f"{r['train_time_sec']:>6.1f}s {r['params']:>10,}")

    print("-" * 100)
    best = sorted_results[0]
    print(f"\n最佳配置: {best['name']}")
    print(f"  学习率: {best['lr']}, 批量大小: {best['batch_size']}")
    print(f"  验证准确率: {best['val_accuracy']:.4f} ({best['val_accuracy']*100:.2f}%)")

    return best


# ============================================================
# Main — 分阶段调优
# ============================================================
def main():
    print("=" * 80)
    print("CNN 超参数调优 — 系统搜索")
    print("=" * 80)

    # 固定数据划分
    X_train, y_train_cat, X_val, y_val_cat, y_val = load_fixed_split()
    print(f"\n数据: 训练 {len(X_train)} 张 | 验证 {len(X_val)} 张")
    print(f"训练分布: {np.bincount(np.argmax(y_train_cat, axis=1))}")

    all_results = []

    # ================================================================
    # Phase 1: 学习率搜索（固定架构=baseline_2conv, batch=16）
    # ================================================================
    print("\n\n" + "#" * 70)
    print("# Phase 1: 学习率搜索 (baseline 2Conv, batch=16)")
    print("#" * 70)

    for lr in [0.001, 0.0005, 0.0001, 0.00005]:
        name = f"LR={lr}"
        result, model, hist = run_experiment(
            name, build_baseline_2conv, lr, 16,
            X_train, y_train_cat, X_val, y_val_cat, y_val)
        all_results.append(result)

    # 找最佳 LR
    phase1_best = max(all_results[-4:], key=lambda r: r['val_accuracy'])
    best_lr = phase1_best['lr']
    print(f"\n>>> Phase 1 最佳 LR = {best_lr} (ValAcc={phase1_best['val_accuracy']:.4f})")

    # ================================================================
    # Phase 2: 批量大小搜索（固定架构=baseline, lr=best_lr）
    # ================================================================
    print("\n\n" + "#" * 70)
    print(f"# Phase 2: 批量大小搜索 (baseline 2Conv, LR={best_lr})")
    print("#" * 70)

    for bs in [8, 32, 64]:
        name = f"BS={bs}_LR={best_lr}"
        result, model, hist = run_experiment(
            name, build_baseline_2conv, best_lr, bs,
            X_train, y_train_cat, X_val, y_val_cat, y_val)
        all_results.append(result)

    phase2_candidates = [r for r in all_results if abs(r['lr'] - best_lr) < 1e-9 and r['batch_size'] in [8, 16, 32, 64]]
    if phase2_candidates:
        phase2_best = max(phase2_candidates, key=lambda r: r['val_accuracy'])
        best_bs = phase2_best['batch_size']
        print(f"\n>>> Phase 2 最佳 Batch = {best_bs} (ValAcc={phase2_best['val_accuracy']:.4f})")
    else:
        best_bs = 16

    # ================================================================
    # Phase 3: 架构搜索（更深 + BN，使用最佳 LR 和 Batch）
    # ================================================================
    print("\n\n" + "#" * 70)
    print(f"# Phase 3: 架构搜索 (LR={best_lr}, BS={best_bs})")
    print("#" * 70)

    architectures = [
        # (名称, 构建函数)
        ("3Conv_BN", build_deeper_3conv),
        ("4Conv_BN", build_deep_4conv),
        ("2Conv_BN", build_medium_2conv_bn),
        # 也试一组更强的 Dropout
        ("2Conv_StrongDrop", lambda: build_baseline_2conv(dropout1=0.4, dropout2=0.5, dropout_fc=0.6)),
        # 更强的 L2
        ("2Conv_StrongL2", lambda: build_baseline_2conv(l2=5e-4)),
    ]

    for arch_name, build_fn in architectures:
        name = f"{arch_name}_LR{best_lr}_BS{best_bs}"
        result, model, hist = run_experiment(
            name, build_fn, best_lr, best_bs,
            X_train, y_train_cat, X_val, y_val_cat, y_val)
        all_results.append(result)

    # ================================================================
    # 最终汇总
    # ================================================================
    best = print_summary(all_results)

    # 保存所有结果
    results_path = os.path.join(OUTPUT_DIR, 'tuning_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n完整结果已保存: {results_path}")

    # ================================================================
    # 用最佳配置重新训练并保存最终模型
    # ================================================================
    print("\n\n" + "=" * 70)
    print("用最佳配置重新训练最终模型...")
    print("=" * 70)

    # 根据最佳配置选择 build_fn
    arch_map = {
        '2Conv_Baseline': build_baseline_2conv,
        '3Conv_BN': build_deeper_3conv,
        '4Conv_BN': build_deep_4conv,
        '2Conv_BN': build_medium_2conv_bn,
    }

    # 解析最佳配置
    best_name_parts = best['name'].split('_')
    if '3Conv_BN' in best['name']:
        final_build = build_deeper_3conv
    elif '4Conv_BN' in best['name']:
        final_build = build_deep_4conv
    elif '2Conv_BN' in best['name']:
        final_build = build_medium_2conv_bn
    elif 'StrongDrop' in best['name']:
        final_build = lambda: build_baseline_2conv(dropout1=0.4, dropout2=0.5, dropout_fc=0.6)
    elif 'StrongL2' in best['name']:
        final_build = lambda: build_baseline_2conv(l2=5e-4)
    else:
        final_build = build_baseline_2conv

    # 重新训练并保存
    tf.random.set_seed(RANDOM_SEED)
    final_result, final_model, final_history = run_experiment(
        f"FINAL_{best['name']}", final_build, best['lr'], best['batch_size'],
        X_train, y_train_cat, X_val, y_val_cat, y_val, epochs=100)

    # 保存模型
    final_model.save(os.path.join(OUTPUT_DIR, 'best_tuned_model.keras'))

    # 在验证集上做完整评估
    print(f"\n{'='*60}")
    print("最佳模型详细评估")
    print("=" * 60)

    from sklearn.metrics import classification_report, confusion_matrix

    y_probs = final_model.predict(X_val, batch_size=best['batch_size'], verbose=0)
    y_pred = np.argmax(y_probs, axis=1)

    print(f"\n整体准确率: {final_result['val_accuracy']:.4f} ({final_result['val_accuracy']*100:.2f}%)")
    print(f"\n分类报告:")
    print(classification_report(y_val, y_pred, digits=4,
                                target_names=[f'数字{i}' for i in range(10)], zero_division=0))

    # 保存最终报告
    cm = confusion_matrix(y_val, y_pred)
    per_class = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)

    final_report = {
        'best_config': best,
        'final_val_accuracy': float(final_result['val_accuracy']),
        'per_class_accuracy': {str(i): float(a) for i, a in enumerate(per_class)},
        'confusion_matrix': cm.tolist(),
        'all_tuning_results': all_results,
    }
    with open(os.path.join(OUTPUT_DIR, 'final_report.json'), 'w', encoding='utf-8') as f:
        json.dump(final_report, f, indent=2, ensure_ascii=False)

    print(f"\n最终报告: {OUTPUT_DIR}/final_report.json")
    print(f"最佳模型: {OUTPUT_DIR}/best_tuned_model.keras")
    print(f"\n{'='*60}")
    print("调优完成!")
    print(f"  最佳验证准确率: {final_result['val_accuracy']*100:.2f}%")


if __name__ == '__main__':
    main()
