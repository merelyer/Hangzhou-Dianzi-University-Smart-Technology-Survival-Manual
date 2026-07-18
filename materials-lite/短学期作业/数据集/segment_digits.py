"""
Digit Segmentation Pipeline (Unicode-safe)
Reads images 1.jpg and 2.jpg, segments individual handwritten digits,
and saves them as separate image files for labeling.

Approach:
1. Convert to grayscale
2. Remove grid lines using morphological operations
3. Find contours of individual digits
4. Filter contours by size, aspect ratio, and shape
5. Sort by position (top-to-bottom, left-to-right)
6. Extract and save each digit with padding
7. Generate a label template CSV
"""

import cv2
import numpy as np
import os
import csv

# ============ CONFIGURATION ============
INPUT_DIR = r"C:\Users\aa257\Desktop\复习\短学期作业\数据集"
OUTPUT_DIR = os.path.join(INPUT_DIR, "segmented_digits")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Filtering parameters
MIN_CONTOUR_AREA = 150       # Min pixel area for a digit
MAX_CONTOUR_AREA = 10000     # Max pixel area
MIN_ASPECT_RATIO = 0.15      # Width/Height min
MAX_ASPECT_RATIO = 6.0       # Width/Height max
MIN_DIM = 8                  # Min width or height in pixels
PADDING = 12                  # Padding around extracted digit

# Grid line removal
HORIZONTAL_KERNEL_RATIO = 0.012
VERTICAL_KERNEL_RATIO = 0.012

# Row grouping threshold (pixels)
ROW_THRESHOLD = 50


# ============ UNICODE-SAFE I/O HELPERS ============

def imread_unicode(filepath):
    """Read image from path with Unicode characters (cv2.imread fails on Windows)."""
    with open(filepath, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img


def imwrite_unicode(filepath, img):
    """Write image to path with Unicode characters (cv2.imwrite fails on Windows)."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext == '.png':
        success, encoded = cv2.imencode('.png', img)
    elif ext in ('.jpg', '.jpeg'):
        success, encoded = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    else:
        success, encoded = cv2.imencode(ext, img)
    if success:
        with open(filepath, 'wb') as f:
            f.write(encoded.tobytes())
        return True
    return False


# ============ PROCESSING FUNCTIONS ============

def remove_grid_lines(gray):
    """
    Remove grid lines using morphological operations.
    Grid lines are thin, long, straight lines; digits are thicker and irregular.
    """
    h, w = gray.shape
    h_kernel_size = max(25, int(w * HORIZONTAL_KERNEL_RATIO))
    v_kernel_size = max(25, int(h * VERTICAL_KERNEL_RATIO))

    # Binarize
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Detect horizontal lines
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_size, 1))
    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)

    # Detect vertical lines
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_size))
    v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)

    # Combine and dilate
    grid = cv2.add(h_lines, v_lines)
    grid_dilated = cv2.dilate(grid, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)

    # Subtract grid
    cleaned = cv2.subtract(binary, grid_dilated)

    # Morph close to reconnect split digits
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close, iterations=1)

    return cleaned, binary


def extract_digits(cleaned_binary, gray_image, image_label):
    """Find, filter, sort, and extract individual digit images."""
    contours, _ = cv2.findContours(cleaned_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    digit_boxes = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)

        if area < MIN_CONTOUR_AREA or area > MAX_CONTOUR_AREA:
            continue
        if w < MIN_DIM or h < MIN_DIM:
            continue
        aspect = w / h if h > 0 else 0
        if aspect < MIN_ASPECT_RATIO or aspect > MAX_ASPECT_RATIO:
            continue
        # Exclude horizontal line fragments
        if w > 4 * h and h < 18:
            continue
        # Exclude vertical line fragments
        if h > 4 * w and w < 18:
            continue

        digit_boxes.append((x, y, w, h, area))

    print(f"  {len(contours)} contours -> {len(digit_boxes)} digits kept")

    if not digit_boxes:
        return [], []

    # Group into rows by y-coordinate
    digit_boxes.sort(key=lambda b: b[1])
    rows = []
    current_row = [digit_boxes[0]]
    row_y = digit_boxes[0][1] + digit_boxes[0][3] / 2

    for box in digit_boxes[1:]:
        y_center = box[1] + box[3] / 2
        if abs(y_center - row_y) < ROW_THRESHOLD:
            current_row.append(box)
        else:
            rows.append(sorted(current_row, key=lambda b: b[0]))
            current_row = [box]
            row_y = y_center
    if current_row:
        rows.append(sorted(current_row, key=lambda b: b[0]))

    # Build ordered list
    ordered = []
    for ri, row in enumerate(rows):
        for ci, box in enumerate(row):
            ordered.append({
                'row': ri, 'col': ci,
                'x': box[0], 'y': box[1], 'w': box[2], 'h': box[3],
                'area': int(box[4]),
            })

    print(f"  {len(rows)} rows, {len(ordered)} digits")

    # Extract images
    gh, gw = gray_image.shape
    digit_images = []
    for d in ordered:
        x1 = max(0, d['x'] - PADDING)
        y1 = max(0, d['y'] - PADDING)
        x2 = min(gw, d['x'] + d['w'] + PADDING)
        y2 = min(gh, d['y'] + d['h'] + PADDING)
        digit_images.append(gray_image[y1:y2, x1:x2])

    return digit_images, ordered


def save_digits(digit_images, ordered, output_dir, img_label):
    """Save each digit as PNG."""
    saved = []
    for i, (dim, od) in enumerate(zip(digit_images, ordered)):
        fname = f"img{img_label}_r{od['row']:03d}_c{od['col']:03d}.png"
        fpath = os.path.join(output_dir, fname)
        imwrite_unicode(fpath, dim)
        saved.append({
            'filename': fname, 'path': fpath,
            'row': od['row'], 'col': od['col'],
            'x': od['x'], 'y': od['y'], 'w': od['w'], 'h': od['h'],
            'area': od['area'],
        })
    print(f"  Saved {len(saved)} images")
    return saved


def save_debug_image(gray, ordered, output_path):
    """Draw bounding boxes for visual verification."""
    debug = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    colors = [(0,255,0),(255,0,0),(0,0,255),(255,255,0),(255,0,255),
              (0,255,255),(128,255,0),(0,128,255),(255,0,128),(128,0,255)]
    for i, d in enumerate(ordered):
        color = colors[d['row'] % len(colors)]
        cv2.rectangle(debug, (d['x'], d['y']), (d['x']+d['w'], d['y']+d['h']), color, 2)
        if i % 5 == 0:  # Label every 5th to reduce clutter
            cv2.putText(debug, str(i), (d['x'], d['y']-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    imwrite_unicode(output_path, debug)
    print(f"  Debug image saved")


def process_image(filename, img_label):
    """Full pipeline for one image."""
    print(f"\n{'='*60}")
    print(f"Processing: {filename}")
    print(f"{'='*60}")

    filepath = os.path.join(INPUT_DIR, filename)
    img = imread_unicode(filepath)
    if img is None:
        print(f"  ERROR: Cannot read {filename}")
        return []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(f"  Size: {gray.shape[1]}x{gray.shape[0]}")

    # Grid removal + binarization
    cleaned, binary = remove_grid_lines(gray)

    # Save debug intermediates
    debug_dir = os.path.join(OUTPUT_DIR, "debug")
    os.makedirs(debug_dir, exist_ok=True)
    imwrite_unicode(os.path.join(debug_dir, f"{img_label}_binary.png"), binary)
    imwrite_unicode(os.path.join(debug_dir, f"{img_label}_cleaned.png"), cleaned)

    # Extract digits
    digit_imgs, ordered = extract_digits(cleaned, gray, img_label)
    if not digit_imgs:
        return []

    # Save
    saved = save_digits(digit_imgs, ordered, OUTPUT_DIR, img_label)

    # Debug viz
    save_debug_image(gray, ordered, os.path.join(debug_dir, f"{img_label}_boxes.png"))

    return saved


def create_label_csv(all_saved, output_dir):
    """Generate CSV template for manual labeling."""
    csv_path = os.path.join(output_dir, "labels.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'img_source', 'row', 'col', 'x', 'y', 'w', 'h', 'area', 'digit_label'])
        for item in all_saved:
            writer.writerow([
                item['filename'],
                item['filename'][3],  # '1' or '2' from "img1_..." or "img2_..."
                item['row'], item['col'],
                item['x'], item['y'], item['w'], item['h'],
                item['area'], ''
            ])
    print(f"\nLabel template: {csv_path}")
    return csv_path


# ============ MAIN ============

def main():
    print("=" * 60)
    print("DIGIT SEGMENTATION PIPELINE")
    print("=" * 60)

    all_saved = []
    for fname, label in [('1.jpg', '1'), ('2.jpg', '2')]:
        saved = process_image(fname, label)
        all_saved.extend(saved)

    csv_path = create_label_csv(all_saved, OUTPUT_DIR)

    print(f"\n{'='*60}")
    print(f"DONE — {len(all_saved)} digits extracted")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Labels:  {csv_path}")
    print(f"Fill in the 'digit_label' column (0-9) for each image.")


if __name__ == '__main__':
    main()
