"""
数字标注工具 v2 — 按键盘 0-9 快速标注手写数字

============================================================
操作说明：
  0-9      : 标注当前数字
  Backspace: 回退到上一个
  Space    : 跳过（不标注）
  S        : 保存进度
  Esc      : 保存并退出
  Enter    : 下一个（不标注）
============================================================
"""

import cv2
import numpy as np
import os
import csv
import json

INPUT_DIR = r"C:\Users\aa257\Desktop\复习\短学期作业\数据集\segmented_digits_v3"
LABELS_CSV = os.path.join(INPUT_DIR, "labels.csv")
PROGRESS_FILE = os.path.join(INPUT_DIR, "labeling_progress.json")
DISPLAY_SIZE = 200  # 放大显示尺寸
WINDOW_NAME = "数字标注 - 按0-9标注"

def imread_u(path):
    with open(path, 'rb') as f:
        return cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_GRAYSCALE)

def load_labels(csv_path):
    labels = {}
    if os.path.exists(csv_path):
        with open(csv_path, 'r', encoding='utf-8-sig') as f:
            for row in csv.DictReader(f):
                val = row.get('digit_label', '').strip()
                if val in '0123456789':
                    labels[row['filename']] = val
    return labels

def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f).get('last_index', 0)
    return 0

def save_progress(index):
    with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
        json.dump({'last_index': index}, f)

def save_csv(all_filenames, labels_dict):
    original_rows = {}
    if os.path.exists(LABELS_CSV):
        with open(LABELS_CSV, 'r', encoding='utf-8-sig') as f:
            for row in csv.DictReader(f):
                original_rows[row['filename']] = row
    first = next(iter(original_rows.values())) if original_rows else {'filename':'','img_source':'','row':'','col':'','x':'','y':'','w':'','h':'','area':'','digit_label':''}
    fieldnames = list(first.keys())

    with open(LABELS_CSV, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(fieldnames)
        for fn in all_filenames:
            row = []
            for fn2 in fieldnames:
                if fn2 == 'digit_label':
                    row.append(labels_dict.get(fn, ''))
                else:
                    row.append(original_rows.get(fn, {}).get(fn2, ''))
            w.writerow(row)

def make_display(digit_img):
    """放大数字图片并添加边框"""
    h, w = digit_img.shape
    scale = DISPLAY_SIZE / max(h, w, 1)
    nw, nh = max(30, int(w*scale)), max(30, int(h*scale))
    resized = cv2.resize(digit_img, (nw, nh), interpolation=cv2.INTER_CUBIC)
    # 白色画布
    canvas_h, canvas_w = DISPLAY_SIZE + 80, max(DISPLAY_SIZE + 80, 380)
    canvas = np.full((canvas_h, canvas_w), 255, dtype=np.uint8)
    # 画一个浅灰边框
    cv2.rectangle(canvas, (10, 10), (canvas_w-10, DISPLAY_SIZE+45), (220, 220, 220), 1)
    # 居中放置数字
    y_off = (DISPLAY_SIZE + 45 - nh) // 2
    x_off = (canvas_w - nw) // 2
    canvas[y_off:y_off+nh, x_off:x_off+nw] = resized
    return cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

def show_start_screen(total, already):
    """显示启动引导画面"""
    canvas = np.full((400, 550, 3), 255, dtype=np.uint8)
    lines = [
        ("数字标注工具", 0.9, (0, 100, 0)),
        ("", 0.5, (0,0,0)),
        (f"共 {total} 张图片，已标注 {already} 张", 0.6, (80, 80, 80)),
        ("", 0.5, (0,0,0)),
        ("操作方法:", 0.7, (0, 0, 180)),
        ("  按 0-9 键  → 标注当前数字", 0.55, (50, 50, 50)),
        ("  按 Backspace → 回退修改", 0.55, (50, 50, 50)),
        ("  按 Space    → 跳过（不标注）", 0.55, (50, 50, 50)),
        ("  按 S        → 手动保存", 0.55, (50, 50, 50)),
        ("  按 Esc      → 保存并退出", 0.55, (50, 50, 50)),
        ("", 0.5, (0,0,0)),
        ("请点击此窗口，然后按 Enter 开始", 0.65, (200, 50, 50)),
    ]
    y = 35
    for text, scale, color in lines:
        if text:
            cv2.putText(canvas, text, (30, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2 if scale>0.7 else 1)
        y += int(32 * scale) + 8

    cv2.imshow(WINDOW_NAME, canvas)
    # 等待用户点击窗口并按回车
    while True:
        k = cv2.waitKey(100) & 0xFF
        if k == 13 or k == 32:  # Enter or Space
            return True
        if k == 27 or k == ord('q'):  # Esc
            return False

def main():
    print("=" * 50)
    print("数字标注工具 v2")
    print("=" * 50)

    if not os.path.exists(LABELS_CSV):
        print("错误: 找不到 labels.csv，请先运行 segment_v3.py")
        return

    filenames = []
    with open(LABELS_CSV, 'r', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            filenames.append(row['filename'])

    total = len(filenames)
    labels = load_labels(LABELS_CSV)
    idx = load_progress()

    print(f"共 {total} 张图片，已标注 {len(labels)} 张")
    if idx > 0:
        print(f"从第 {idx+1} 张继续...")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 600, 480)

    # 显示启动画面
    if not show_start_screen(total, len(labels)):
        cv2.destroyAllWindows()
        print("已取消。")
        return

    save_counter = 0

    while 0 <= idx < total:
        fname = filenames[idx]
        digit = imread_u(os.path.join(INPUT_DIR, fname))
        if digit is None:
            idx += 1
            continue

        display = make_display(digit)
        cur = labels.get(fname, '')

        # 文字叠加
        cv2.putText(display, f"{idx+1} / {total}", (15, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 80, 80), 1)
        if cur:
            cv2.putText(display, f"已标注: {cur}", (15, 52),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 150, 0), 2)
        else:
            cv2.putText(display, "未标注", (15, 52),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 220), 2)
        cv2.putText(display, f"已标: {len(labels)}张", (15, 76),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)
        cv2.putText(display, f"文件: {fname}", (15, display.shape[0]-14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 160, 160), 1)
        # 底部按键提示
        hint_bar = np.full((32, display.shape[1], 3), 245, dtype=np.uint8)
        cv2.putText(hint_bar, "[0-9]标注 [Back]回退 [Space]跳过 [S]保存 [Esc]退出",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 100, 100), 1)
        display = np.vstack([display, hint_bar])

        cv2.imshow(WINDOW_NAME, display)
        key = cv2.waitKey(0) & 0xFF

        if key == 27:  # Esc
            break
        elif key == ord('s'):
            save_csv(filenames, labels)
            save_progress(idx)
            save_counter = 0
            print(f"  [已保存] {len(labels)} 个标注 (第{idx+1}张)")
        elif key == 8:  # Backspace
            if cur:
                del labels[fname]
            idx = max(0, idx - 1)
        elif key == 32:  # Space
            idx += 1
        elif key == 13:  # Enter
            idx += 1
        elif ord('0') <= key <= ord('9'):
            labels[fname] = chr(key)
            idx += 1
            save_counter += 1
            if save_counter % 20 == 0:
                print(f"  已标注 {len(labels)}/{total}")
        else:
            # 未知按键 — 忽略并提示
            if key > 0:
                print(f"  [忽略] 未知按键: {key}, 请按 0-9 标注")

        # 自动保存
        if save_counter >= 50:
            save_csv(filenames, labels)
            save_progress(idx)
            print(f"  [自动保存] {len(labels)} 个标注")
            save_counter = 0

    cv2.destroyAllWindows()
    save_csv(filenames, labels)
    save_progress(idx)
    print(f"\n{'='*50}")
    print(f"完成! 已标注 {len(labels)} / {total} 张")
    print(f"结果保存在: {LABELS_CSV}")
    print(f"下次运行 'python label_tool.py' 可从第 {idx+1} 张继续")


if __name__ == '__main__':
    main()
