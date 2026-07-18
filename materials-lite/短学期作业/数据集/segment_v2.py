"""
Improved Digit Segmentation Pipeline v2
Uses adaptive thresholding for better handling of handwriting,
and grid-aware processing to improve digit detection.
"""

import cv2
import numpy as np
import os
import csv

# ============ CONFIGURATION ============
INPUT_DIR = r"C:\Users\aa257\Desktop\复习\短学期作业\数据集"
OUTPUT_DIR = os.path.join(INPUT_DIR, "segmented_digits_v2")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Filtering
MIN_AREA = 120
MAX_AREA = 12000
MIN_DIM = 8
PADDING = 12
ROW_GAP = 55  # Max y-distance between rows

# ============ HELPERS ============

def imread_u(path):
    with open(path, 'rb') as f:
        return cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)

def imwrite_u(path, img):
    ext = os.path.splitext(path)[1]
    ok, enc = cv2.imencode(ext, img)
    if ok:
        with open(path, 'wb') as f:
            f.write(enc.tobytes())

# ============ PIPELINE ============

def preprocess(gray):
    """
    Use adaptive thresholding which handles varying ink density better than Otsu.
    Also try to suppress grid lines using a morphological approach.
    """
    # Step 1: Enhance contrast with CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Step 2: Adaptive threshold — handles local variations
    binary = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, blockSize=31, C=8
    )

    # Step 3: Remove thin grid lines while preserving thick digit strokes
    # Use morphological opening with a small kernel to detect thin lines
    h, w = binary.shape
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, int(w * 0.01)), 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, int(h * 0.01))))

    # Detect both directions
    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
    v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)

    # Dilate grid mask slightly
    grid = cv2.add(h_lines, v_lines)
    grid = cv2.dilate(grid, np.ones((2, 2), np.uint8))

    # Remove grid
    cleaned = cv2.subtract(binary, grid)

    # Close small gaps in digits
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))

    return cleaned, binary, enhanced


def find_digits(binary, min_area=MIN_AREA, max_area=MAX_AREA):
    """Find connected components that look like digits."""
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)

        if area < min_area or area > max_area:
            continue
        if w < MIN_DIM or h < MIN_DIM:
            continue

        # Aspect ratio check
        ar = w / max(h, 1)
        if ar < 0.12 or ar > 7:
            continue

        # Very thin horizontal/vertical (grid line remnants)
        if w > 3 * h and h < 18:
            continue
        if h > 3 * w and w < 18:
            continue

        boxes.append((x, y, w, h, int(area)))

    return boxes


def group_into_rows(boxes):
    """Sort boxes by position: top-to-bottom, then left-to-right within rows."""
    if not boxes:
        return []

    boxes.sort(key=lambda b: b[1])  # sort by y

    rows = []
    current = [boxes[0]]
    row_y = boxes[0][1] + boxes[0][3] / 2

    for b in boxes[1:]:
        yc = b[1] + b[3] / 2
        if abs(yc - row_y) < ROW_GAP:
            current.append(b)
        else:
            rows.append(sorted(current, key=lambda x: x[0]))  # sort row by x
            current = [b]
            row_y = yc
    if current:
        rows.append(sorted(current, key=lambda x: x[0]))

    # Build ordered list
    result = []
    for ri, row in enumerate(rows):
        for ci, (x, y, w, h, area) in enumerate(row):
            result.append({
                'row': ri, 'col': ci,
                'x': x, 'y': y, 'w': w, 'h': h, 'area': area,
            })
    return result, rows


def extract_and_save(gray, ordered, output_dir, img_label):
    """Extract digit images from gray image and save."""
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
    """Draw colored bounding boxes."""
    debug = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    colors = [(0,255,0),(255,0,0),(0,0,255),(255,255,0),(255,0,255),
              (0,255,255),(128,255,0),(0,128,255),(255,0,128),(128,0,255)]
    for i, d in enumerate(ordered):
        c = colors[d['row'] % len(colors)]
        cv2.rectangle(debug, (d['x'], d['y']), (d['x']+d['w'], d['y']+d['h']), c, 2)
    imwrite_u(path, debug)


def process_image(filename, img_label):
    """Full pipeline."""
    print(f"\n{'='*60}")
    print(f"Processing: {filename}")
    print(f"{'='*60}")

    img = imread_u(os.path.join(INPUT_DIR, filename))
    if img is None:
        print(f"  ERROR: Cannot read {filename}")
        return []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(f"  Size: {gray.shape[1]}x{gray.shape[0]}")

    # Preprocess
    cleaned, binary, enhanced = preprocess(gray)

    # Save debug intermediates
    dbg = os.path.join(OUTPUT_DIR, "debug")
    os.makedirs(dbg, exist_ok=True)
    imwrite_u(os.path.join(dbg, f"{img_label}_enhanced.png"), enhanced)
    imwrite_u(os.path.join(dbg, f"{img_label}_binary_adaptive.png"), binary)
    imwrite_u(os.path.join(dbg, f"{img_label}_cleaned.png"), cleaned)

    # Find digits
    boxes = find_digits(cleaned)
    print(f"  Found {len(boxes)} candidate digit boxes")

    # Try also on original binary (in case cleaning removed too much)
    boxes_raw = find_digits(binary)
    print(f"  Found {len(boxes_raw)} from raw binary")

    # Use the better set (more contours usually means better segmentation)
    use_boxes = boxes if len(boxes) >= len(boxes_raw) else boxes_raw
    print(f"  Using set with {len(use_boxes)} boxes")

    # Group into rows and sort
    ordered, rows = group_into_rows(use_boxes)
    print(f"  {len(rows)} rows, {len(ordered)} digits")

    if not ordered:
        return []

    # Save digits
    saved = extract_and_save(gray, ordered, OUTPUT_DIR, img_label)

    # Debug visualization
    draw_debug(gray, ordered, os.path.join(dbg, f"{img_label}_boxes.png"))

    return saved


def main():
    print("=" * 60)
    print("DIGIT SEGMENTATION v2 (Adaptive Threshold)")
    print("=" * 60)

    all_saved = []
    for fname, label in [('1.jpg', '1'), ('2.jpg', '2')]:
        saved = process_image(fname, label)
        all_saved.extend(saved)

    # Create labels CSV
    csv_path = os.path.join(OUTPUT_DIR, "labels.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['filename', 'img_source', 'row', 'col', 'x', 'y', 'w', 'h', 'area', 'digit_label'])
        for s in all_saved:
            w.writerow([s['filename'], s['src'], s['row'], s['col'],
                        s['x'], s['y'], s['w'], s['h'], s['area'], ''])

    print(f"\n{'='*60}")
    print(f"DONE — {len(all_saved)} digits extracted")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Labels:  {csv_path}")


if __name__ == '__main__':
    main()
