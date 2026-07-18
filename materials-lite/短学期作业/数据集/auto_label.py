"""
自动数字识别器
基于图像特征（孔洞数、形状、投影等）自动识别手写数字 0-9
使用决策树规则 + 置信度评分
"""

import cv2
import numpy as np
import os
import csv
from collections import Counter

INPUT_DIR = r"C:\Users\aa257\Desktop\复习\短学期作业\数据集\segmented_digits_v3"
LABELS_CSV = os.path.join(INPUT_DIR, "labels.csv")

def imread_u(path):
    with open(path, 'rb') as f:
        return cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_GRAYSCALE)

def count_holes(binary):
    """
    计算数字中的孔洞数。
    方法：对背景进行floodfill，未被填充的白色区域即为孔洞。
    """
    h, w = binary.shape
    # 在图像边缘加一圈白色边框
    padded = np.pad(binary, ((1,1),(1,1)), 'constant', constant_values=0)

    # Flood fill from corner (background)
    mask = np.zeros((h+4, w+4), np.uint8)
    cv2.floodFill(padded.copy(), mask, (0, 0), 128)

    # 未被填充的白色区域 = 孔洞
    remaining = (padded == 255) & (mask[1:-1, 1:-1] == 0)

    # 用连通组件分析统计孔洞数
    num_labels, labels = cv2.connectedComponents(remaining.astype(np.uint8))
    holes = num_labels - 1  # 减去背景

    # 过滤掉太小的孔洞（噪声）
    real_holes = 0
    for i in range(1, num_labels):
        if np.sum(labels == i) > 20:  # 至少20个像素
            real_holes += 1

    return real_holes, remaining


def extract_features(gray):
    """提取数字图像的所有特征"""
    features = {}

    # 二值化
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    h, w = binary.shape

    # 裁剪到实际数字区域
    coords = cv2.findNonZero(binary)
    if coords is None:
        return None
    x, y, bw, bh = cv2.boundingRect(coords)

    # 基本几何特征
    features['width'] = bw
    features['height'] = bh
    features['aspect_ratio'] = bw / max(bh, 1)
    features['area'] = cv2.countNonZero(binary)
    features['extent'] = features['area'] / max(bw * bh, 1)
    features['hull_area'] = cv2.contourArea(cv2.convexHull(coords)) if len(coords) > 2 else 0
    features['solidity'] = features['area'] / max(features['hull_area'], 1)

    # 孔洞数
    holes, hole_mask = count_holes(binary)
    features['holes'] = holes

    # 投影特征
    h_proj = np.sum(binary, axis=1)  # 水平投影
    v_proj = np.sum(binary, axis=0)  # 垂直投影

    features['h_proj_max'] = np.max(h_proj) if len(h_proj) > 0 else 0
    features['v_proj_max'] = np.max(v_proj) if len(v_proj) > 0 else 0

    # 上/下半部分密度比
    mid_y = bh // 2
    features['top_density'] = np.sum(h_proj[:mid_y]) / max(features['area'], 1)
    features['bottom_density'] = np.sum(h_proj[mid_y:]) / max(features['area'], 1)

    # 左/右半部分密度比
    mid_x = bw // 2
    features['left_density'] = np.sum(v_proj[:mid_x]) / max(features['area'], 1)
    features['right_density'] = np.sum(v_proj[mid_x:]) / max(features['area'], 1)

    # Hu矩（形状描述子）
    moments = cv2.moments(binary)
    hu = cv2.HuMoments(moments)
    for i in range(7):
        features[f'hu{i+1}'] = -np.sign(hu[i]) * np.log10(max(abs(hu[i]), 1e-10))

    # 水平投影的峰值数量（用于区分2和3等）
    if len(h_proj) > 1:
        smoothed = np.convolve(h_proj, np.ones(5)/5, mode='same')
        peaks = 0
        for i in range(1, len(smoothed) - 1):
            if smoothed[i] > smoothed[i-1] and smoothed[i] > smoothed[i+1]:
                if smoothed[i] > np.max(smoothed) * 0.2:
                    peaks += 1
        features['h_peaks'] = peaks
    else:
        features['h_peaks'] = 0

    # 垂直投影峰值数
    if len(v_proj) > 1:
        smoothed = np.convolve(v_proj, np.ones(5)/5, mode='same')
        v_peaks = 0
        for i in range(1, len(smoothed) - 1):
            if smoothed[i] > smoothed[i-1] and smoothed[i] > smoothed[i+1]:
                if smoothed[i] > np.max(smoothed) * 0.2:
                    v_peaks += 1
        features['v_peaks'] = v_peaks
    else:
        features['v_peaks'] = 0

    # 笔画宽度估计
    # 对binary进行距离变换
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    features['stroke_width'] = np.mean(dist[dist > 0]) * 2 if np.any(dist > 0) else 0

    # 上1/3与下1/3密度比
    third_y = bh // 3
    features['top_third'] = np.sum(h_proj[:third_y]) / max(features['area'], 1)
    features['mid_third'] = np.sum(h_proj[third_y:2*third_y]) / max(features['area'], 1)
    features['bot_third'] = np.sum(h_proj[2*third_y:]) / max(features['area'], 1)

    return features


def classify_digit(features):
    """
    基于特征的决策树分类器。
    返回 (digit, confidence)
    confidence: 0.0-1.0, 越高越可信
    """
    if features is None:
        return ('?', 0.0)

    holes = features['holes']
    ar = features['aspect_ratio']
    top = features['top_density']
    bot = features['bottom_density']
    left = features['left_density']
    right = features['right_density']
    h_peaks = features['h_peaks']
    v_peaks = features['v_peaks']
    solidity = features['solidity']
    extent = features['extent']
    area = features['area']
    top3 = features['top_third']
    mid3 = features['mid_third']
    bot3 = features['bot_third']
    sw = features['stroke_width']

    # ===== 孔洞分类 =====
    if holes == 2:
        return ('8', 0.95)

    if holes == 1:
        # 0: 孔洞居中，对称
        # 6: 墨迹集中在左/下
        # 9: 墨迹集中在右/上
        # 4: 孔洞小且偏上

        if top3 > 0.45 and bot3 < 0.30:
            # 墨迹在上，孔洞也在上 → 9
            return ('9', 0.75)
        elif bot3 > 0.45 and top3 < 0.30:
            # 墨迹在下，孔洞在上 → 6
            return ('6', 0.75)
        elif abs(top3 - bot3) < 0.12 and 0.7 < ar < 1.4:
            # 对称分布，接近圆形 → 0
            return ('0', 0.82)
        elif solidity < 0.65:
            # 不太实心 → 可能是4
            return ('4', 0.55)
        else:
            # 默认按密度分布判断
            if top3 < bot3:
                return ('6', 0.55)
            else:
                return ('9', 0.55)

    if holes == 0:
        # 1: 非常窄
        if ar < 0.45 and extent < 0.5:
            return ('1', 0.85)
        if ar < 0.35:
            return ('1', 0.90)

        # 7: 顶部重，对角线
        if top > 0.6 and bot < 0.25 and left > 0.45:
            return ('7', 0.65)

        # 2: 底部水平笔画 + 顶部曲线
        if bot > 0.45 and h_peaks >= 2 and ar > 0.6:
            return ('2', 0.55)

        # 3: 两个水平峰（上下两个圆弧）
        if h_peaks >= 2 and 0.5 < ar < 1.3 and abs(top3 - bot3) < 0.2:
            return ('3', 0.55)

        # 5: 顶部水平条 + 底部圆弧
        if top > 0.4 and bot > 0.35 and h_peaks >= 1:
            return ('5', 0.50)

        # 4: 三角形，左密度集中
        if solidity < 0.6 and left > 0.5:
            return ('4', 0.50)

        # 2: 底部重，有弧度
        if bot > 0.4 and ar > 0.55:
            return ('2', 0.50)

        # 3: 比较对称的两个弧
        if 0.6 < ar < 1.3 and h_peaks >= 1:
            return ('3', 0.45)

        # 7: 倾斜的
        if top > 0.55 and left > 0.5:
            return ('7', 0.50)

        # 1: 窄长
        if ar < 0.55:
            return ('1', 0.60)

        # 兜底：按形状猜测
        if ar < 0.6:
            return ('1', 0.30)
        elif bot > top:
            return ('2', 0.25)
        else:
            return ('5', 0.20)

    # 无法判断
    return ('?', 0.0)


def auto_label_all():
    """对所有分割出的数字进行自动识别"""
    print("=" * 60)
    print("自动数字识别器")
    print("=" * 60)

    # 读取文件列表
    filenames = []
    with open(LABELS_CSV, 'r', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            filenames.append(row['filename'])

    total = len(filenames)
    print(f"共 {total} 张图片待识别...\n")

    results = {}
    stats = Counter()
    confidences = []

    for i, fname in enumerate(filenames):
        img = imread_u(os.path.join(INPUT_DIR, fname))
        if img is None:
            continue

        features = extract_features(img)
        digit, conf = classify_digit(features)
        results[fname] = digit
        stats[digit] += 1
        confidences.append(conf)

        if (i + 1) % 100 == 0:
            print(f"  进度: {i+1}/{total}")

    # 保存结果到CSV
    save_results(results, filenames)

    # 统计报告
    print(f"\n{'='*60}")
    print("识别完成！统计：")
    print(f"{'='*60}")
    for d in '0123456789?':
        cnt = stats.get(d, 0)
        bar = '█' * (cnt * 40 // max(total, 1))
        print(f"  数字 {d}: {cnt:4d} 个  {bar}")

    avg_conf = np.mean(confidences) if confidences else 0
    high_conf = sum(1 for c in confidences if c >= 0.7)
    medium_conf = sum(1 for c in confidences if 0.4 <= c < 0.7)
    low_conf = sum(1 for c in confidences if c < 0.4)

    print(f"\n置信度分布:")
    print(f"  高置信度 (≥70%): {high_conf} 张 ({100*high_conf/max(total,1):.0f}%)")
    print(f"  中置信度 (40-70%): {medium_conf} 张 ({100*medium_conf/max(total,1):.0f}%)")
    print(f"  低置信度 (<40%): {low_conf} 张 ({100*low_conf/max(total,1):.0f}%)")
    print(f"  平均置信度: {avg_conf:.1%}")

    uncertain = stats.get('?', 0)
    if uncertain > 0:
        print(f"\n  ⚠ {uncertain} 张未能识别（标记为'?'），建议人工复核")

    print(f"\n结果已保存到: {LABELS_CSV}")


def save_results(labels_dict, filenames):
    """保存标注结果"""
    original_rows = {}
    if os.path.exists(LABELS_CSV):
        with open(LABELS_CSV, 'r', encoding='utf-8-sig') as f:
            for row in csv.DictReader(f):
                original_rows[row['filename']] = row

    first = next(iter(original_rows.values())) if original_rows else {}
    fieldnames = list(first.keys()) if first else ['filename','img_source','row','col','x','y','w','h','area','digit_label']

    with open(LABELS_CSV, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(fieldnames)
        for fn in filenames:
            row = []
            for key in fieldnames:
                if key == 'digit_label':
                    row.append(labels_dict.get(fn, ''))
                else:
                    row.append(original_rows.get(fn, {}).get(key, ''))
            w.writerow(row)


if __name__ == '__main__':
    auto_label_all()
