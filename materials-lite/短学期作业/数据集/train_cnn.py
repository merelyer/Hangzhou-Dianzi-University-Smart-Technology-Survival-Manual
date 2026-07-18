"""
CNN 手写数字识别 — 训练脚本
架构：2 个卷积层 + 1 个全连接层（符合课程要求的简单 CNN）
数据集：增强后的 digits_augmented.npz（5000 张，10 类均衡）
"""

import numpy as np
import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ============================================================
# 配置
# ============================================================
DATA_PATH = r"C:\Users\aa257\Desktop\复习\短学期作业\数据集\digits_augmented\digits_augmented.npz"
OUTPUT_DIR = r"C:\Users\aa257\Desktop\复习\短学期作业\数据集\cnn_model"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_SEED = 42

# ============================================================
# 1. 加载数据
# ============================================================
def load_data():
    print("=" * 60)
    print("加载增强数据集...")
    data = np.load(DATA_PATH)
    X = data['images']       # (5000, 28, 28) float32 [0, 1]
    y = data['labels']       # (5000,) int32

    print(f"  X shape: {X.shape}, dtype: {X.dtype}")
    print(f"  y shape: {y.shape}")
    print(f"  类别分布: {np.bincount(y)}")

    # 添加通道维度: (N, 28, 28) → (N, 28, 28, 1)
    X = np.expand_dims(X, axis=-1)

    # One-hot 编码
    y_cat = keras.utils.to_categorical(y, 10)

    return X, y, y_cat


def split_data(X, y, y_cat):
    """分层抽样划分为训练集、验证集、测试集"""
    from sklearn.model_selection import train_test_split

    # 先分出测试集
    X_temp, X_test, y_temp, y_test, y_cat_temp, y_cat_test = train_test_split(
        X, y, y_cat,
        test_size=TEST_SPLIT,
        stratify=y,
        random_state=RANDOM_SEED,
    )

    # 再从剩余中分出验证集
    val_ratio = VALIDATION_SPLIT / (1 - TEST_SPLIT)
    X_train, X_val, y_train, y_val, y_cat_train, y_cat_val = train_test_split(
        X_temp, y_temp, y_cat_temp,
        test_size=val_ratio,
        stratify=y_temp,
        random_state=RANDOM_SEED,
    )

    print(f"\n数据划分:")
    print(f"  训练集: {X_train.shape[0]} 张")
    print(f"  验证集: {X_val.shape[0]} 张")
    print(f"  测试集: {X_test.shape[0]} 张")
    print(f"  训练集类别分布: {np.bincount(y_train)}")

    return (X_train, y_train, y_cat_train), (X_val, y_val, y_cat_val), (X_test, y_test, y_cat_test)


# ============================================================
# 2. CNN 模型架构
# ============================================================
def build_cnn(input_shape=(28, 28, 1), num_classes=10):
    """
    ============================================================
    CNN 架构设计
    ============================================================

    【卷积层 1】
      - 32 个 3×3 卷积核，stride=1，same padding
      - 激活函数：ReLU（避免梯度消失，计算高效）
      - 输出尺寸：28×28×32

    【池化层 1】
      - 2×2 最大池化，stride=2
      - 作用：降采样，保留主要特征，减少参数
      - 输出尺寸：14×14×32

    【卷积层 2】
      - 64 个 3×3 卷积核，stride=1，same padding
      - 激活函数：ReLU
      - 输出尺寸：14×14×64

    【池化层 2】
      - 2×2 最大池化，stride=2
      - 输出尺寸：7×7×64

    【全连接层】
      - Flatten: 7×7×64 = 3136 → 128
      - 128 个神经元，ReLU 激活
      - Dropout(0.5)：防止过拟合
      - 输出层: 128 → 10，Softmax 激活

    【输出层】
      - 10 个神经元（对应 0-9）
      - Softmax 激活函数 → 输出各类别概率分布 ∑p=1
    """
    model = keras.Sequential(name="DigitCNN")

    # ---- 卷积块 1 ----
    model.add(layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same',
        activation='relu',
        input_shape=input_shape,
        name='Conv2D_1'
    ))
    model.add(layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        name='MaxPool_1'
    ))

    # ---- 卷积块 2 ----
    model.add(layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same',
        activation='relu',
        name='Conv2D_2'
    ))
    model.add(layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        name='MaxPool_2'
    ))

    # ---- 全连接层 ----
    model.add(layers.Flatten(name='Flatten'))
    model.add(layers.Dense(
        units=128,
        activation='relu',
        name='FC_128'
    ))
    model.add(layers.Dropout(rate=0.5, name='Dropout'))

    # ---- 输出层 ----
    model.add(layers.Dense(
        units=num_classes,
        activation='softmax',
        name='Output'
    ))

    return model


# ============================================================
# 3. 模型编译
# ============================================================
def compile_model(model):
    """
    编译配置：

    【损失函数】Categorical Crossentropy（多分类交叉熵）
      - 适用于 one-hot 编码的多分类问题
      - 衡量预测概率分布与真实分布的差异

    【优化器】Adam（自适应矩估计）
      - 结合 Momentum 和 RMSProp 的优点
      - 自适应学习率，收敛快且稳定
      - 适合大多数深度学习任务

    【评估指标】Accuracy（准确率）
      - 预测正确的样本数 / 总样本数
      - 同时监控 Top-1 和 Top-2 准确率
    """
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


# ============================================================
# 4. 训练
# ============================================================
def train_model(model, X_train, y_cat_train, X_val, y_cat_val):
    """训练模型，使用早停和学习率衰减"""
    callbacks = [
        # 早停：验证损失不再下降时停止
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
        # 学习率衰减：平台期降低学习率
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1,
        ),
        # 模型检查点
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(OUTPUT_DIR, 'best_model.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1,
        ),
    ]

    print("\n" + "=" * 60)
    print("开始训练...")
    print("=" * 60)

    history = model.fit(
        X_train, y_cat_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_cat_val),
        callbacks=callbacks,
        verbose=1,
    )

    return history


# ============================================================
# 5. 评估
# ============================================================
def evaluate_model(model, X_test, y_test, y_cat_test, history):
    """在测试集上评估模型，输出详细报告"""
    print("\n" + "=" * 60)
    print("测试集评估")
    print("=" * 60)

    # 整体评估
    test_loss, test_acc = model.evaluate(X_test, y_cat_test, verbose=0)
    print(f"\n  测试集准确率: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"  测试集损失:   {test_loss:.4f}")

    # 逐类评估
    from sklearn.metrics import classification_report, confusion_matrix

    y_pred_probs = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print("\n--- 分类报告 ---")
    print(classification_report(
        y_test, y_pred,
        digits=4,
        target_names=[f'数字{i}' for i in range(10)]
    ))

    print("--- 混淆矩阵 ---")
    cm = confusion_matrix(y_test, y_pred)
    # 格式化输出
    header = "      " + " ".join(f" 预{i}" for i in range(10))
    print(header)
    for i, row in enumerate(cm):
        print(f"  真{i}: " + " ".join(f"{v:4d}" for v in row))

    # 准确率 = 对角线之和 / 总数
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    print("\n--- 各类别准确率 ---")
    for i, acc in enumerate(per_class_acc):
        bar = '█' * int(acc * 40)
        print(f"  数字 {i}: {acc:.4f} ({acc*100:5.1f}%) {bar}")

    return test_acc, cm


# ============================================================
# 6. 可视化
# ============================================================
def plot_training(history):
    """绘制训练曲线"""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 损失曲线
    axes[0].plot(history.history['loss'], label='训练损失', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='验证损失', linewidth=2)
    axes[0].set_title('模型损失曲线', fontsize=14)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 准确率曲线
    axes[1].plot(history.history['accuracy'], label='训练准确率', linewidth=2)
    axes[1].plot(history.history['val_accuracy'], label='验证准确率', linewidth=2)
    axes[1].set_title('模型准确率曲线', fontsize=14)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'training_curves.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n训练曲线已保存: {save_path}")


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 60)
    print("CNN 手写数字识别 — 训练与评估")
    print("=" * 60)

    # 1. 加载数据
    X, y, y_cat = load_data()

    # 2. 划分数据集
    (X_train, y_train, y_cat_train), \
        (X_val, y_val, y_cat_val), \
        (X_test, y_test, y_cat_test) = split_data(X, y, y_cat)

    # 3. 构建模型
    print("\n" + "=" * 60)
    print("构建 CNN 模型...")
    model = build_cnn()
    model.summary()

    # 4. 编译
    compile_model(model)

    # 5. 训练
    history = train_model(model, X_train, y_cat_train, X_val, y_cat_val)

    # 6. 评估
    test_acc, cm = evaluate_model(model, X_test, y_test, y_cat_test, history)

    # 7. 可视化
    try:
        plot_training(history)
    except Exception as e:
        print(f"\n  [跳过] 绘图失败: {e}")

    # 8. 保存模型
    final_path = os.path.join(OUTPUT_DIR, 'digit_cnn_final.keras')
    model.save(final_path)
    print(f"\n模型已保存: {final_path}")

    # 9. 输出总结
    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)
    print(f"  测试准确率: {test_acc*100:.2f}%")
    print(f"  模型文件:   {OUTPUT_DIR}")
    print(f"  CNN 架构:   2×Conv + 2×Pool + 1×FC(128) + Output(10)")
    print(f"  总参数数:    {model.count_params():,}")


if __name__ == '__main__':
    main()
