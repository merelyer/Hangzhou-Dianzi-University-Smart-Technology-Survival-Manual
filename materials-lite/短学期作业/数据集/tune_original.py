"""
对原始增强分割图像模型进行系统超参数调优
Phase 1: 学习率 (0.0005, 0.001, 0.002, 0.005)
Phase 2: 批量大小 (8, 16, 32)
Phase 3: 架构对比 (更深+BN+Dropout变体)
Phase 4: 最佳配置 4500 测试集评估
"""

import os, cv2, json, time, numpy as np
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ============================================================
INPUT_DIR = r"C:\Users\aa257\Desktop\复习\短学期作业\数据集\原始增强分割图像"
OUTPUT_DIR = r"C:\Users\aa257\Desktop\复习\短学期作业\数据集\original_tuning"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_PER_CLASS = 40
VAL_PER_CLASS = 10
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# ============================================================
# 数据加载（固定划分）
# ============================================================
def imread_u(path):
    with open(path, 'rb') as f:
        return cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_GRAYSCALE)

def preprocess(gray):
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(binary)
    if coords is None:
        return np.zeros((28,28), dtype=np.uint8)
    x, y, w, h = cv2.boundingRect(coords)
    pad = 3
    x1, y1 = max(0,x-pad), max(0,y-pad)
    x2, y2 = min(binary.shape[1],x+w+pad), min(binary.shape[0],y+h+pad)
    digit = binary[y1:y2, x1:x2]
    scale = 20.0 / max(digit.shape[0], digit.shape[1], 1)
    nh, nw = max(1,int(digit.shape[0]*scale)), max(1,int(digit.shape[1]*scale))
    resized = cv2.resize(digit, (nw,nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((28,28), dtype=np.uint8)
    off_y, off_x = (28-nh)//2, (28-nw)//2
    canvas[off_y:off_y+nh, off_x:off_x+nw] = resized
    return canvas

def load_fixed_split():
    X_all, y_all = [], []
    for d in range(10):
        folder = os.path.join(INPUT_DIR, str(d))
        for f in os.listdir(folder):
            img = imread_u(os.path.join(folder, f))
            if img is not None:
                X_all.append(preprocess(img).astype(np.float32)/255.0)
                y_all.append(d)
    X_all, y_all = np.array(X_all), np.array(y_all, dtype=np.int32)

    class_idx = {d: np.where(y_all == d)[0] for d in range(10)}
    train_idx, val_idx = [], []
    for d in range(10):
        idx = class_idx[d]
        np.random.shuffle(idx)
        train_idx.extend(idx[:TRAIN_PER_CLASS].tolist())
        val_idx.extend(idx[TRAIN_PER_CLASS:TRAIN_PER_CLASS+VAL_PER_CLASS].tolist())

    X_train = np.expand_dims(X_all[train_idx], -1)
    y_train_cat = keras.utils.to_categorical(y_all[train_idx], 10)
    X_val = np.expand_dims(X_all[val_idx], -1)
    y_val_cat = keras.utils.to_categorical(y_all[val_idx], 10)
    y_val = y_all[val_idx]
    return X_train, y_train_cat, X_val, y_val_cat, y_val


# ============================================================
# 模型构建器
# ============================================================
def build_baseline():
    """当前最佳: 2Conv + L2 + Dropout"""
    m = keras.Sequential(name='Baseline')
    m.add(layers.Conv2D(32,(3,3),padding='same',activation='relu',
          kernel_regularizer=keras.regularizers.l2(1e-4),input_shape=(28,28,1)))
    m.add(layers.MaxPooling2D((2,2))); m.add(layers.Dropout(0.25))
    m.add(layers.Conv2D(64,(3,3),padding='same',activation='relu',
          kernel_regularizer=keras.regularizers.l2(1e-4)))
    m.add(layers.MaxPooling2D((2,2))); m.add(layers.Dropout(0.3))
    m.add(layers.Flatten())
    m.add(layers.Dense(128,activation='relu',kernel_regularizer=keras.regularizers.l2(1e-4)))
    m.add(layers.Dropout(0.5))
    m.add(layers.Dense(10,activation='softmax'))
    return m

def build_deeper():
    """3Conv + BN + 更强Dropout"""
    m = keras.Sequential(name='Deeper_BN')
    m.add(layers.Conv2D(32,(3,3),padding='same',use_bias=False,input_shape=(28,28,1)))
    m.add(layers.BatchNormalization()); m.add(layers.Activation('relu'))
    m.add(layers.Conv2D(32,(3,3),padding='same',use_bias=False))
    m.add(layers.BatchNormalization()); m.add(layers.Activation('relu'))
    m.add(layers.MaxPooling2D((2,2))); m.add(layers.Dropout(0.25))

    m.add(layers.Conv2D(64,(3,3),padding='same',use_bias=False))
    m.add(layers.BatchNormalization()); m.add(layers.Activation('relu'))
    m.add(layers.Conv2D(64,(3,3),padding='same',use_bias=False))
    m.add(layers.BatchNormalization()); m.add(layers.Activation('relu'))
    m.add(layers.MaxPooling2D((2,2))); m.add(layers.Dropout(0.35))

    m.add(layers.Conv2D(128,(3,3),padding='same',use_bias=False))
    m.add(layers.BatchNormalization()); m.add(layers.Activation('relu'))
    m.add(layers.MaxPooling2D((2,2))); m.add(layers.Dropout(0.4))

    m.add(layers.Flatten())
    m.add(layers.Dense(128,use_bias=False,kernel_regularizer=keras.regularizers.l2(1e-4)))
    m.add(layers.BatchNormalization()); m.add(layers.Activation('relu'))
    m.add(layers.Dropout(0.5))
    m.add(layers.Dense(10,activation='softmax'))
    return m

def build_medium_bn():
    """2Conv + BN — 中等"""
    m = keras.Sequential(name='Medium_BN')
    m.add(layers.Conv2D(32,(3,3),padding='same',use_bias=False,input_shape=(28,28,1)))
    m.add(layers.BatchNormalization()); m.add(layers.Activation('relu'))
    m.add(layers.MaxPooling2D((2,2))); m.add(layers.Dropout(0.25))
    m.add(layers.Conv2D(64,(3,3),padding='same',use_bias=False))
    m.add(layers.BatchNormalization()); m.add(layers.Activation('relu'))
    m.add(layers.MaxPooling2D((2,2))); m.add(layers.Dropout(0.35))
    m.add(layers.Flatten())
    m.add(layers.Dense(128,use_bias=False,kernel_regularizer=keras.regularizers.l2(1e-4)))
    m.add(layers.BatchNormalization()); m.add(layers.Activation('relu'))
    m.add(layers.Dropout(0.5))
    m.add(layers.Dense(10,activation='softmax'))
    return m

def build_strong_reg():
    """Baseline + 强正则化"""
    m = keras.Sequential(name='StrongReg')
    m.add(layers.Conv2D(32,(3,3),padding='same',activation='relu',
          kernel_regularizer=keras.regularizers.l2(5e-4),input_shape=(28,28,1)))
    m.add(layers.MaxPooling2D((2,2))); m.add(layers.Dropout(0.35))
    m.add(layers.Conv2D(64,(3,3),padding='same',activation='relu',
          kernel_regularizer=keras.regularizers.l2(5e-4)))
    m.add(layers.MaxPooling2D((2,2))); m.add(layers.Dropout(0.45))
    m.add(layers.Flatten())
    m.add(layers.Dense(128,activation='relu',kernel_regularizer=keras.regularizers.l2(5e-4)))
    m.add(layers.Dropout(0.6))
    m.add(layers.Dense(10,activation='softmax'))
    return m

def build_wider():
    """2Conv 更宽 (64→128)"""
    m = keras.Sequential(name='Wider')
    m.add(layers.Conv2D(48,(3,3),padding='same',activation='relu',
          kernel_regularizer=keras.regularizers.l2(1e-4),input_shape=(28,28,1)))
    m.add(layers.MaxPooling2D((2,2))); m.add(layers.Dropout(0.25))
    m.add(layers.Conv2D(96,(3,3),padding='same',activation='relu',
          kernel_regularizer=keras.regularizers.l2(1e-4)))
    m.add(layers.MaxPooling2D((2,2))); m.add(layers.Dropout(0.3))
    m.add(layers.Flatten())
    m.add(layers.Dense(192,activation='relu',kernel_regularizer=keras.regularizers.l2(1e-4)))
    m.add(layers.Dropout(0.5))
    m.add(layers.Dense(10,activation='softmax'))
    return m


# ============================================================
# 实验运行
# ============================================================
def run_exp(name, build_fn, lr, bs, X_train, y_train_cat, X_val, y_val_cat, y_val, epochs=80):
    print('\n[EXP] %s | LR=%.5f BS=%d' % (name, lr, bs))
    tf.random.set_seed(RANDOM_SEED)
    model = build_fn()
    model.compile(optimizer=keras.optimizers.Adam(lr),
                  loss='categorical_crossentropy', metrics=['accuracy'])

    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=20, restore_best_weights=True, verbose=0)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6, verbose=0)

    augmenter = keras.Sequential([
        layers.RandomRotation(0.04, fill_mode='constant', fill_value=0),
        layers.RandomTranslation(0.12, 0.12, fill_mode='constant', fill_value=0),
        layers.RandomZoom(0.15, 0.15, fill_mode='constant', fill_value=0),
        layers.RandomContrast(0.25),
    ])

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train_cat))
    train_ds = train_ds.shuffle(512).batch(bs)
    train_ds = train_ds.map(lambda x, y: (augmenter(x, training=True), y),
                            num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val_cat))
    val_ds = val_ds.batch(bs).prefetch(tf.data.AUTOTUNE)

    t0 = time.time()
    history = model.fit(train_ds, epochs=epochs, validation_data=val_ds,
                        callbacks=[early_stop, reduce_lr], verbose=0)
    elapsed = time.time() - t0

    y_pred = np.argmax(model.predict(val_ds, verbose=0), axis=1)
    from sklearn.metrics import accuracy_score
    val_acc = accuracy_score(y_val, y_pred)

    result = {
        'name': name, 'lr': lr, 'batch_size': bs,
        'val_acc': float(val_acc),
        'best_val_acc': float(max(history.history['val_accuracy'])),
        'best_train_acc': float(max(history.history['accuracy'])),
        'epochs': len(history.history['loss']),
        'time': float(elapsed),
        'params': int(model.count_params()),
    }
    print('  ValAcc=%.4f BestVal=%.4f Epochs=%d Time=%.1fs Params=%d' % (
        result['val_acc'], result['best_val_acc'], result['epochs'], elapsed, result['params']))
    return result, model


# ============================================================
def main():
    print("=" * 70)
    print("原始增强数据 CNN 超参数调优")
    print("=" * 70)

    X_train, y_train_cat, X_val, y_val_cat, y_val = load_fixed_split()
    print("Data: train=%d val=%d" % (len(X_train), len(X_val)))

    all_results = []

    # Phase 1: 学习率
    print("\n" + "#" * 60)
    print("# Phase 1: Learning Rate (Baseline, BS=16)")
    print("#" * 60)
    for lr in [0.0005, 0.001, 0.002, 0.003]:
        r, _ = run_exp("LR=%.4f" % lr, build_baseline, lr, 16,
                       X_train, y_train_cat, X_val, y_val_cat, y_val)
        all_results.append(r)
    best_lr = max([r for r in all_results], key=lambda r: r['val_acc'])['lr']
    print("\n>>> Best LR = %.4f" % best_lr)

    # Phase 2: 批量大小
    print("\n" + "#" * 60)
    print("# Phase 2: Batch Size (Baseline, LR=%.4f)" % best_lr)
    print("#" * 60)
    for bs in [8, 32, 64]:
        r, _ = run_exp("BS=%d" % bs, build_baseline, best_lr, bs,
                       X_train, y_train_cat, X_val, y_val_cat, y_val)
        all_results.append(r)
    candidates = [r for r in all_results if abs(r['lr']-best_lr) < 1e-8]
    candidates += [r for r in all_results if 'BS=' in r['name']]
    best_bs = max(candidates, key=lambda r: r['val_acc'])['batch_size']
    print("\n>>> Best BS = %d" % best_bs)

    # Phase 3: 架构
    print("\n" + "#" * 60)
    print("# Phase 3: Architecture (LR=%.4f, BS=%d)" % (best_lr, best_bs))
    print("#" * 60)
    archs = [
        ("Deeper+BN", build_deeper),
        ("Medium+BN", build_medium_bn),
        ("StrongReg", build_strong_reg),
        ("Wider", build_wider),
    ]
    for name, fn in archs:
        r, _ = run_exp(name, fn, best_lr, best_bs,
                       X_train, y_train_cat, X_val, y_val_cat, y_val)
        all_results.append(r)

    # ---- 汇总排名 ----
    print("\n\n" + "=" * 70)
    print("TUNING RESULTS (sorted by val_acc)")
    print("=" * 70)
    print("%-22s %8s %5s %8s %8s %7s %7s %10s" % (
        'Experiment', 'LR', 'BS', 'ValAcc', 'BestVal', 'Epochs', 'Time', 'Params'))
    print("-" * 85)
    sorted_r = sorted(all_results, key=lambda r: r['val_acc'], reverse=True)
    for i, r in enumerate(sorted_r):
        tag = '>>>' if i == 0 else '   '
        print("%s %-18s %8.4f %5d %8.4f %8.4f %7d %6.1fs %10d" % (
            tag, r['name'], r['lr'], r['batch_size'],
            r['val_acc'], r['best_val_acc'], r['epochs'], r['time'], r['params']))

    best = sorted_r[0]
    print("\n>>> BEST: %s (LR=%.4f, BS=%d, ValAcc=%.4f)" % (
        best['name'], best['lr'], best['batch_size'], best['val_acc']))

    # 保存
    with open(os.path.join(OUTPUT_DIR, 'tuning_results.json'), 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # ================================================================
    # Phase 4: 最佳模型在 4500 测试集评估
    # ================================================================
    print("\n\n" + "=" * 70)
    print("Phase 4: Best model on 4500 test set")
    print("=" * 70)

    # 重新训练最佳模型
    arch_map = {
        'LR=': build_baseline, 'BS=': build_baseline,
        'Deeper+BN': build_deeper, 'Medium+BN': build_medium_bn,
        'StrongReg': build_strong_reg, 'Wider': build_wider,
    }
    best_build = build_baseline
    for key, fn in arch_map.items():
        if key in best['name']:
            best_build = fn
            break

    tf.random.set_seed(RANDOM_SEED)
    _, best_model = run_exp("FINAL_"+best['name'], best_build, best['lr'], best['batch_size'],
                            X_train, y_train_cat, X_val, y_val_cat, y_val, epochs=100)
    best_model.save(os.path.join(OUTPUT_DIR, 'best_tuned_model.keras'))

    # 4500 测试集评估
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

    X_all, y_all = [], []
    for d in range(10):
        folder = os.path.join(INPUT_DIR, str(d))
        for f in sorted(os.listdir(folder)):
            img = imread_u(os.path.join(folder, f))
            if img is not None:
                X_all.append(preprocess(img).astype(np.float32)/255.0)
                y_all.append(d)
    X_all = np.expand_dims(np.array(X_all), -1)
    y_all = np.array(y_all, dtype=np.int32)

    class_idx = {d: np.where(y_all == d)[0] for d in range(10)}
    test_idx = []
    for d in range(10):
        idx = class_idx[d]
        np.random.shuffle(idx)
        test_idx.extend(idx[50:].tolist())

    X_test = X_all[test_idx]; y_test = y_all[test_idx]
    print("Test set: %d samples" % len(X_test))
    print("Test distribution: %s" % str(np.bincount(y_test)))

    y_pred = np.argmax(best_model.predict(X_test, batch_size=64, verbose=0), axis=1)
    test_acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    per_class = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)

    print("\nTest Accuracy: %.4f (%.2f%%)" % (test_acc, test_acc*100))
    print("\nPer-class:")
    for i, p in enumerate(per_class):
        bar = '#' * int(p*50)
        print("  %d: %.4f (%.1f%%) %s" % (i, p, p*100, bar))
    print("Macro avg: %.4f" % per_class.mean())

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4,
          target_names=['%d'%i for i in range(10)], zero_division=0))

    print("\nConfusion Matrix (rows=true, cols=pred):")
    print("      " + " ".join("%4d" % i for i in range(10)))
    for i, row in enumerate(cm):
        print("  %d: " % i + " ".join("%4d" % v for v in row))

    report = {
        'best_config': best,
        'test_accuracy': float(test_acc),
        'per_class_accuracy': {str(i): float(a) for i, a in enumerate(per_class)},
    }
    with open(os.path.join(OUTPUT_DIR, 'final_report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("DONE! Test Accuracy: %.2f%%" % (test_acc*100))
    print("Output: %s" % OUTPUT_DIR)
    print("Best model: best_tuned_model.keras")
    print("=" * 70)


if __name__ == '__main__':
    main()
