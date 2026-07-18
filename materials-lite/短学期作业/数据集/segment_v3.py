"""
改进版数字分割 v3 — 更精细的参数调优
解决 v2 中行合并和误检问题。
"""

import cv2
import numpy as np
import os
import csv

INPUT_DIR = r"C:\Users\aa257\Desktop\复习\短学期作业\数据集"
OUTPUT_DIR = os.path.join(INPUT_DIR, "segmented_digits_v3")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 调优后的参数
MIN_AREA = 150
MAX_AREA = 6000        # 更严格，减少合并数字
MIN_DIM = 8
PADDING = 12
ROW_GAP = 38           # 更紧凑的行间距，减少行合并

def imread_u(path):
    with open(path, 'rb') as f:
        return cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)

def imwrite_u(path, img):
    ext = os.path.splitext(path)[1]
    ok, enc = cv2.imencode(ext, img)
    if ok:
        with open(path, 'wb') as f:
            f.write(enc.tobytes())

def preprocess(gray):
    """CLAHE + 自适应阈值 + 网格线去除"""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    binary = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, blockSize=31, C=8
    )

    h, w = binary.shape
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, int(w * 0.01)), 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, int(h * 0.01))))

    grid = cv2.add(
        cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel),
        cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)
    )
    grid = cv2.dilate(grid, np.ones((2, 2), np.uint8))
    cleaned = cv2.subtract(binary, grid)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))

    return cleaned, binary, enhanced

def find_digits(binary):
    """查找数字轮廓"""
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        if area < MIN_AREA or area > MAX_AREA:
            continue
        if w < MIN_DIM or h < MIN_DIM:
            continue
        ar = w / max(h, 1)
        if ar < 0.12 or ar > 7:
            continue
        if w > 3 * h and h < 18:
            continue
        if h > 3 * w and w < 18:
            continue
        # 额外：排除接近正方形的极小区域（噪声）
        if area < 250 and 0.8 < ar < 1.2:
            continue
        boxes.append((x, y, w, h, int(area)))
    return boxes

def group_into_rows(boxes):
    """按行分组并排序"""
    if not boxes:
        return [], []
    boxes.sort(key=lambda b: b[1])
    rows = []
    current = [boxes[0]]
    row_y = boxes[0][1] + boxes[0][3] / 2
    for b in boxes[1:]:
        yc = b[1] + b[3] / 2
        if abs(yc - row_y) < ROW_GAP:
            current.append(b)
        else:
            rows.append(sorted(current, key=lambda x: x[0]))
            current = [b]
            row_y = yc
    if current:
        rows.append(sorted(current, key=lambda x: x[0]))

    result = []
    for ri, row in enumerate(rows):
        for ci, (x, y, w, h, area) in enumerate(row):
            result.append({'row': ri, 'col': ci, 'x': x, 'y': y, 'w': w, 'h': h, 'area': area})
    return result, rows

def extract_and_save(gray, ordered, output_dir, img_label):
    """提取并保存每个数字图像"""
    gh, gw = gray.shape
    saved = []
    for d in ordered:
        x1 = max(0, d['x'] - PADDING)
        y1 = max(0, d['y'] - PADDING)
        x2 = min(gw, d['x'] + d['w'] + PADDING)
        y2 = min(gh, d['y'] + d['h'] + PADDING)
        digit = gray[y1:y2, x1:x2]
        fname = f"img{img_label}_r{d['row']:03d}_c{d['col']:03d}.png"
        fpath = os.path.join(output_dir, fname)
        imwrite_u(fpath, digit)
        saved.append({
            'filename': fname, 'path': fpath,
            'src': img_label, 'row': d['row'], 'col': d['col'],
            'x': d['x'], 'y': d['y'], 'w': d['w'], 'h': d['h'], 'area': d['area'],
        })
    return saved

def draw_debug(gray, ordered, path):
    """绘制调试用的边界框图像"""
    debug = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    colors = [(0,255,0),(255,0,0),(0,0,255),(255,255,0),(255,0,255),
              (0,255,255),(128,255,0),(0,128,255),(255,0,128),(128,0,255)]
    for d in ordered:
        c = colors[d['row'] % len(colors)]
        cv2.rectangle(debug, (d['x'], d['y']), (d['x']+d['w'], d['y']+d['h']), c, 2)
    imwrite_u(path, debug)

def process_image(filename, img_label):
    """处理单张图像"""
    print(f"\n{'='*60}")
    print(f"处理: {filename}")
    img = imread_u(os.path.join(INPUT_DIR, filename))
    if img is None:
        return []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(f"  尺寸: {gray.shape[1]}x{gray.shape[0]}")

    cleaned, binary, enhanced = preprocess(gray)

    dbg = os.path.join(OUTPUT_DIR, "debug")
    os.makedirs(dbg, exist_ok=True)
    imwrite_u(os.path.join(dbg, f"{img_label}_cleaned.png"), cleaned)

    boxes = find_digits(cleaned)
    boxes_raw = find_digits(binary)
    use_boxes = boxes if len(boxes) >= len(boxes_raw) else boxes_raw
    print(f"  检测到 {len(use_boxes)} 个数字候选框")

    ordered, rows = group_into_rows(use_boxes)
    print(f"  分为 {len(rows)} 行, 共 {len(ordered)} 个数字")

    if not ordered:
        return []

    saved = extract_and_save(gray, ordered, OUTPUT_DIR, img_label)
    draw_debug(gray, ordered, os.path.join(dbg, f"{img_label}_boxes.png"))
    return saved

def main():
    print("=" * 60)
    print("数字分割 v3（优化参数版）")
    print("=" * 60)

    all_saved = []
    for fname, label in [('1.jpg', '1'), ('2.jpg', '2')]:
        saved = process_image(fname, label)
        all_saved.extend(saved)

    csv_path = os.path.join(OUTPUT_DIR, "labels.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['filename', 'img_source', 'row', 'col', 'x', 'y', 'w', 'h', 'area', 'digit_label'])
        for s in all_saved:
            w.writerow([s['filename'], s['src'], s['row'], s['col'],
                        s['x'], s['y'], s['w'], s['h'], s['area'], ''])

    print(f"\n{'='*60}")
    print(f"完成 — 共提取 {len(all_saved)} 个数字")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"标签模板: {csv_path}")

if __name__ == '__main__':
    main()
