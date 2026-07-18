"""
手写数字识别测试工具
用法: test_model.py <数据集路径>
      或拖拽数据集文件夹到本程序

模型: mixed_v2 混合训练最佳模型
"""

import os, sys, cv2, json, numpy as np
from datetime import datetime

# 抑制 TensorFlow 日志
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def imread_u(path, mode='gray'):
    """Unicode-safe 图像读取"""
    try:
        with open(path, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8)
        flag = cv2.IMREAD_GRAYSCALE if mode == 'gray' else cv2.IMREAD_COLOR
        return cv2.imdecode(data, flag)
    except Exception:
        return None


def auto_detect_and_preprocess(img):
    """
    自动检测图像格式并预处理为 28x28 黑底白字二值图
    支持:
      - 已处理的 28x28 黑底白字图 (直接使用)
      - 28x28 白底黑字图 (反转+Otsu)
      - 28x28 RGB 暗底亮字图 (阈值+Otsu)
      - 变尺寸灰度图 (Otsu+居中缩放)
    """
    if img is None:
        return None

    # 转灰度
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    h, w = gray.shape

    # 检测类型: 看像素分布
    light_ratio = (gray > 128).sum() / gray.size
    mean_val = gray.mean()

    if h == 28 and w == 28:
        if light_ratio < 0.3 and mean_val < 50:
            # 已是黑底白字二值图 (如 aug500)
            binary = gray
        elif light_ratio > 0.6 and mean_val > 150:
            # 白底黑字 (如 wx) → 反转+Otsu
            inv = 255 - gray
            _, binary = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            # RGB暗底亮字 (如 friend) → 自适应阈值
            bg = np.percentile(gray, 20)
            if bg < 255:
                stretched = np.clip((gray.astype(np.float32) - bg) / (255 - bg) * 255, 0, 255).astype(np.uint8)
            else:
                stretched = gray
            _, binary = cv2.threshold(stretched, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        # 变尺寸原始图 → Otsu + 居中缩放
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 裁剪数字区域并居中到 28x28
    coords = cv2.findNonZero(binary)
    if coords is None:
        return np.zeros((28, 28), dtype=np.uint8)

    x, y, bw, bh = cv2.boundingRect(coords)
    pad = 3
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(binary.shape[1], x + bw + pad), min(binary.shape[0], y + bh + pad)
    digit = binary[y1:y2, x1:x2]

    if digit.size == 0:
        return np.zeros((28, 28), dtype=np.uint8)

    scale = 20.0 / max(digit.shape[0], digit.shape[1], 1)
    nh, nw = max(1, int(digit.shape[0] * scale)), max(1, int(digit.shape[1] * scale))
    resized = cv2.resize(digit, (nw, nh), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((28, 28), dtype=np.uint8)
    off_y, off_x = (28 - nh) // 2, (28 - nw) // 2
    canvas[off_y:off_y + nh, off_x:off_x + nw] = resized
    return canvas


def load_dataset(folder_path):
    """加载数据集 (0-9 子文件夹)"""
    X, y = [], []
    for digit in range(10):
        subfolder = os.path.join(folder_path, str(digit))
        if not os.path.exists(subfolder):
            print(f"  [WARN] 找不到文件夹: {subfolder}")
            continue
        files = [f for f in os.listdir(subfolder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        if not files:
            print(f"  [WARN] 数字 {digit} 文件夹为空")
            continue
        for fname in sorted(files):
            img = imread_u(os.path.join(subfolder, fname))
            if img is None:
                continue
            processed = auto_detect_and_preprocess(img)
            if processed is None:
                continue
            X.append(processed.astype(np.float32) / 255.0)
            y.append(digit)
        print(f"  数字 {digit}: {len(files)} 张")
    return np.array(X), np.array(y)


def find_model():
    """查找模型文件"""
    candidates = [
        r'C:\Users\aa257\Desktop\复习\短学期作业\数据集\mixed_v2\best_model.keras',
        r'C:\Users\aa257\Desktop\复习\短学期作业\数据集\mixed_training\best_model.keras',
        r'C:\Users\aa257\Desktop\复习\短学期作业\数据集\original_tuning\best_tuned_model.keras',
    ]
    # 也搜索 exe 同目录
    if getattr(sys, 'frozen', False):
        exe_dir = os.path.dirname(sys.executable)
        candidates.insert(0, os.path.join(exe_dir, 'best_model.keras'))

    for path in candidates:
        if os.path.exists(path):
            return path

    # 让用户手动指定
    print("\n找不到模型文件，请拖拽 .keras 文件到这里:")
    path = input().strip().strip('"')
    if os.path.exists(path):
        return path
    return None


def main():
    print("=" * 60)
    print("  手写数字识别测试工具")
    print("  模型: mixed_v2 混合训练")
    print("=" * 60)

    # 获取数据集路径
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    else:
        print("\n请拖拽数据集文件夹到这里 (或输入路径):")
        data_path = input().strip().strip('"')

    if not os.path.isdir(data_path):
        print(f"[ERROR] 路径无效: {data_path}")
        input("\n按 Enter 退出...")
        return

    print(f"\n数据集: {data_path}")

    # 加载数据
    print("\n加载并预处理图像...")
    try:
        X, y = load_dataset(data_path)
    except Exception as e:
        print(f"[ERROR] 加载失败: {e}")
        input("\n按 Enter 退出...")
        return

    if len(X) == 0:
        print("[ERROR] 未加载到任何图像")
        input("\n按 Enter 退出...")
        return

    X = np.expand_dims(X, -1)
    print(f"\n总计: {len(X)} 张图像")
    dist = np.bincount(y, minlength=10)
    print(f"分布: {' '.join(f'{i}:{c}' for i,c in enumerate(dist) if c>0)}")

    # 加载模型
    model_path = find_model()
    if model_path is None:
        print("[ERROR] 找不到模型文件")
        input("\n按 Enter 退出...")
        return

    print(f"\n加载模型: {model_path}")
    model = keras.models.load_model(model_path)
    print(f"模型: {model.name}, 参数: {model.count_params():,}")

    # 预测
    print("\n预测中...")
    y_probs = model.predict(X, batch_size=64, verbose=0)
    y_pred = np.argmax(y_probs, axis=1)

    # 计算准确率
    acc = accuracy_score(y, y_pred)
    n_errors = (y != y_pred).sum()
    cm = confusion_matrix(y, y_pred)
    per_class = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)

    # 输出结果
    print("\n" + "=" * 60)
    print("  测 试 结 果")
    print("=" * 60)
    print(f"\n  总样本数: {len(y)}")
    print(f"  正确: {len(y) - n_errors}  错误: {n_errors}")
    print(f"  准确率: {acc:.4f} ({acc*100:.2f}%)")

    print(f"\n  各类别准确率:")
    for i, (p, count) in enumerate(zip(per_class, dist)):
        if count > 0:
            bar = '#' * int(p * 40)
            print(f"    数字 {i}: {p*100:5.1f}% ({int(p*count)}/{count}) {bar}")

    print(f"\n  分类报告:")
    present_classes = [i for i in range(10) if dist[i] > 0]
    target_names = [str(i) for i in present_classes]
    y_present = [i for i, c in enumerate(dist) if c > 0]
    print(classification_report(
        y, y_pred, digits=4,
        labels=present_classes,
        target_names=target_names,
        zero_division=0
    ))

    print("  混淆矩阵 (行=真实, 列=预测):")
    header = "        " + " ".join(f"{i:4d}" for i in present_classes)
    print(header)
    print("       " + "-" * (5 * len(present_classes)))
    for i in present_classes:
        row_str = " ".join(f"{cm[i, j]:4d}" for j in present_classes)
        print(f"    {i}:  {row_str}")

    # 保存报告
    report_path = os.path.join(os.path.dirname(data_path) if os.path.isdir(os.path.dirname(data_path)) else '.',
                                f'test_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    try:
        report = {
            'dataset': data_path,
            'model': model_path,
            'accuracy': float(acc),
            'n_samples': int(len(y)),
            'n_errors': int(n_errors),
            'per_class_accuracy': {str(i): float(p) for i, p in enumerate(per_class) if dist[i] > 0},
            'distribution': {str(i): int(c) for i, c in enumerate(dist) if c > 0},
            'confusion_matrix': cm.tolist(),
            'timestamp': datetime.now().isoformat(),
        }
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n报告已保存: {report_path}")
    except Exception as e:
        print(f"\n[WARN] 报告保存失败: {e}")

    print("\n" + "=" * 60)
    input("\n按 Enter 退出...")


if __name__ == '__main__':
    main()
