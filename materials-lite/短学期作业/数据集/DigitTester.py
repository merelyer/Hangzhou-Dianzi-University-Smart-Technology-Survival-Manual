"""
手写数字识别系统 — GUI版 (纯 numpy+Pillow，无 TF/OpenCV)
"""

import os, sys, json, threading, numpy as np
from datetime import datetime

# =========== 配置 ===========
STUDENT_ID = "23061612"
STUDENT_NAME = "郭晨帆"

# =========== 轻量图像处理 (Pillow 替代 OpenCV) ===========
from PIL import Image

def imread_auto(path):
    try:
        img = Image.open(path).convert('L')
        return np.array(img, dtype=np.uint8)
    except Exception:
        return None

def preprocess_auto(gray):
    """自动检测格式 → 28x28 黑底白字 float32

    支持多种输入格式:
      - 28x28 黑底白字 (dataset_aug500) → 直接归一化
      - 28x28 白底黑字 (augmented_images) → 反转+Otsu清理
      - JPEG 压缩 28x28 (aug_28x28) → Otsu 提取
      - 变尺寸原始图 (原始增强分割图像) → Otsu + 居中缩放
    """
    if gray is None:
        return None
    h, w = gray.shape

    # ---- 用边缘像素判断背景颜色 ----
    edges = np.concatenate([gray[0, :], gray[-1, :], gray[1:-1, 0], gray[1:-1, -1]])
    edge_mean = float(np.mean(edges))

    # 情况 1: 28x28 黑底白字（边缘<50，纯黑背景）→ 直接归一化
    if h == 28 and w == 28 and edge_mean < 50:
        return gray.astype(np.float32) / 255.0

    # 情况 2: 28x28 白底黑字（边缘>150，白背景）→ 反转 + Otsu 清理
    if h == 28 and w == 28 and edge_mean > 150:
        inv = 255 - gray
        t = _otsu_threshold(inv)
        # Otsu 清理: 亮像素是数字
        binary = ((inv > t) * 255).astype(np.uint8)
        ys, xs = np.where(binary > 0)
        if len(ys) == 0:
            return np.zeros((28, 28), dtype=np.float32)
        x1, y1 = xs.min(), ys.min(); x2, y2 = xs.max() + 1, ys.max() + 1
        pad = 3
        x1, y1 = max(0, x1-pad), max(0, y1-pad)
        x2, y2 = min(w, x2+pad), min(h, y2+pad)
        digit = binary[y1:y2, x1:x2]
        if digit.size == 0:
            return np.zeros((28, 28), dtype=np.float32)
        scale = 20.0 / max(digit.shape[0], digit.shape[1], 1)
        nh, nw = max(1, int(digit.shape[0]*scale)), max(1, int(digit.shape[1]*scale))
        resized = np.array(Image.fromarray(digit).resize((nw, nh), Image.LANCZOS), dtype=np.uint8)
        canvas = np.zeros((28, 28), dtype=np.float32)
        off_y, off_x = (28-nh)//2, (28-nw)//2
        canvas[off_y:off_y+nh, off_x:off_x+nw] = resized.astype(np.float32) / 255.0
        return canvas

    # 情况 3: JPEG 压缩图 / 变尺寸 / 含噪图 → 完整 Otsu 管线
    t = _otsu_threshold(gray)
    if edge_mean > 128:
        # 白底 → 暗像素是数字
        binary = ((gray <= t) * 255).astype(np.uint8)
    else:
        # 黑底 → 亮像素是数字
        binary = ((gray > t) * 255).astype(np.uint8)

    # 找数字区域
    ys, xs = np.where(binary > 0)
    if len(ys) == 0:
        return np.zeros((28, 28), dtype=np.float32)

    x1, y1 = xs.min(), ys.min()
    x2, y2 = xs.max() + 1, ys.max() + 1
    pad = 3
    x1, y1 = max(0, x1-pad), max(0, y1-pad)
    x2, y2 = min(w, x2+pad), min(h, y2+pad)
    digit = binary[y1:y2, x1:x2]

    if digit.size == 0:
        return np.zeros((28, 28), dtype=np.float32)

    # 保持长宽比缩放到 20x20
    scale = 20.0 / max(digit.shape[0], digit.shape[1], 1)
    nh, nw = max(1, int(digit.shape[0]*scale)), max(1, int(digit.shape[1]*scale))
    resized = np.array(Image.fromarray(digit).resize((nw, nh), Image.LANCZOS), dtype=np.uint8)

    canvas = np.zeros((28, 28), dtype=np.float32)
    off_y, off_x = (28-nh)//2, (28-nw)//2
    canvas[off_y:off_y+nh, off_x:off_x+nw] = resized.astype(np.float32) / 255.0
    return canvas

def _otsu_threshold(img):
    """纯 numpy Otsu 阈值"""
    hist = np.bincount(img.ravel(), minlength=256)
    total = img.size
    sum_all = np.dot(np.arange(256), hist)
    w_b = 0; w_f = 0; sum_b = 0; max_var = 0; thr = 128
    for t in range(256):
        w_b += hist[t]
        if w_b == 0: continue
        w_f = total - w_b
        if w_f == 0: break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_all - sum_b) / w_f
        var = w_b * w_f * (m_b - m_f) ** 2
        if var > max_var:
            max_var = var; thr = t
    return thr


# =========== 纯 numpy 评估指标 ===========
def accuracy_score(y_true, y_pred):
    return (np.array(y_true) == np.array(y_pred)).mean()

def confusion_matrix(y_true, y_pred, n_classes=10):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    return cm

def classification_report_str(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    n = max(y_true.max(), y_pred.max()) + 1
    cm = confusion_matrix(y_true, y_pred, n)
    lines = []
    lines.append("              precision    recall  f1-score   support")
    for i in range(n):
        tp = cm[i,i]
        fp = cm[:,i].sum() - tp
        fn = cm[i,:].sum() - tp
        prec = tp/(tp+fp) if (tp+fp)>0 else 0
        rec = tp/(tp+fn) if (tp+fn)>0 else 0
        f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0
        sup = int(cm[i,:].sum())
        if sup > 0:
            lines.append("       %d     %.4f    %.4f    %.4f      %d" % (i, prec, rec, f1, sup))
    total = cm.sum()
    correct = cm.diagonal().sum()
    lines.append("    accuracy                        %.4f      %d" % (correct/total, total))
    return "\n".join(lines)


# =========== 加载 ONNX 模型 ===========
if getattr(sys, 'frozen', False):
    MODEL_PATH = os.path.join(sys._MEIPASS, "model.onnx")
else:
    MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.onnx")
import onnxruntime as ort
ort_session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
ort_input_name = ort_session.get_inputs()[0].name

def predict_batch(images):
    results = []
    for img in images:
        out = ort_session.run(None, {ort_input_name: np.expand_dims(img, 0)})[0]
        results.append((int(np.argmax(out)), float(np.max(out))))
    return results


# =========== GUI ===========
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

class DigitTesterApp:
    def __init__(self, root):
        self.root = root
        self.root.title(f"手写数字识别系统 — {STUDENT_NAME} {STUDENT_ID}")
        self.root.geometry("750x650")
        self.root.resizable(True, True)

        header = tk.Frame(root, bg="#2c3e50", height=80)
        header.pack(fill="x")
        tk.Label(header, text="手写数字识别系统", font=("微软雅黑", 20, "bold"),
                 fg="white", bg="#2c3e50").pack(pady=(10,0))
        tk.Label(header, text=f"学号: {STUDENT_ID}    姓名: {STUDENT_NAME}",
                 font=("微软雅黑", 12), fg="#bdc3c7", bg="#2c3e50").pack(pady=(0,10))

        main = tk.Frame(root, padx=20, pady=15)
        main.pack(fill="both", expand=True)

        # 数据集测试
        ff = tk.LabelFrame(main, text=" 数据集批量测试 ", font=("微软雅黑", 12, "bold"), padx=10, pady=10, fg="#2c3e50")
        ff.pack(fill="x", pady=(0,10))
        br = tk.Frame(ff); br.pack(fill="x")
        self.folder_path = tk.StringVar()
        tk.Entry(br, textvariable=self.folder_path, font=("微软雅黑", 10), state="readonly").pack(side="left", fill="x", expand=True, padx=(0,5))
        tk.Button(br, text="选择文件夹", command=self.select_folder, font=("微软雅黑", 10), bg="#3498db", fg="white", width=12, cursor="hand2").pack(side="right")
        tk.Button(br, text="开始测试", command=self.start_folder_test, font=("微软雅黑", 10), bg="#27ae60", fg="white", width=10, cursor="hand2").pack(side="right", padx=(5,0))

        # 单张识别
        imf = tk.LabelFrame(main, text=" 单张图片识别 ", font=("微软雅黑", 12, "bold"), padx=10, pady=10, fg="#2c3e50")
        imf.pack(fill="x", pady=(0,10))
        ibr = tk.Frame(imf); ibr.pack(fill="x")
        self.img_path = tk.StringVar()
        tk.Entry(ibr, textvariable=self.img_path, font=("微软雅黑", 10), state="readonly").pack(side="left", fill="x", expand=True, padx=(0,5))
        tk.Button(ibr, text="选择图片", command=self.select_image, font=("微软雅黑", 10), bg="#3498db", fg="white", width=12, cursor="hand2").pack(side="right")
        self.single_result = tk.Label(imf, text="识别结果: ---", font=("微软雅黑", 16, "bold"), fg="#e74c3c")
        self.single_result.pack(pady=(10,0))
        self.single_conf = tk.Label(imf, text="", font=("微软雅黑", 10), fg="#7f8c8d")
        self.single_conf.pack()

        self.progress = ttk.Progressbar(main, mode='indeterminate')
        self.status = tk.Label(main, text="就绪", font=("微软雅黑", 9), fg="#95a5a6")
        self.status.pack()

        rf = tk.LabelFrame(main, text=" 测试结果 ", font=("微软雅黑", 12, "bold"), padx=10, pady=10, fg="#2c3e50")
        rf.pack(fill="both", expand=True)
        self.output = scrolledtext.ScrolledText(rf, font=("Consolas", 10), wrap="word", height=18)
        self.output.pack(fill="both", expand=True)
        self.output.tag_configure("title", font=("Consolas", 12, "bold"), foreground="#2c3e50")
        self.output.tag_configure("good", foreground="#27ae60")
        self.output.tag_configure("bad", foreground="#e74c3c")
        self.output.tag_configure("info", foreground="#2980b9")

    def log(self, text, tag=None):
        self.output.insert("end", text + "\n", tag or "")
        self.output.see("end")
        self.root.update_idletasks()

    def select_folder(self):
        p = filedialog.askdirectory(title="选择数据集文件夹（内含0-9子文件夹）")
        if p: self.folder_path.set(p)

    def select_image(self):
        p = filedialog.askopenfilename(title="选择图片", filetypes=[("图片文件", "*.png *.jpg *.jpeg *.bmp")])
        if p:
            self.img_path.set(p)
            self.recognize_single()

    def recognize_single(self):
        path = self.img_path.get()
        if not path: return
        try:
            gray = imread_auto(path)
            if gray is None: self.single_result.config(text="无法读取图片", fg="#e74c3c"); return
            proc = preprocess_auto(gray)
            if proc is None: self.single_result.config(text="预处理失败", fg="#e74c3c"); return
            preds = predict_batch([np.expand_dims(proc, -1)])
            digit, conf = preds[0]
            self.single_result.config(text=f"识别结果: {digit}", fg="#27ae60")
            self.single_conf.config(text=f"置信度: {conf*100:.1f}%")
        except Exception as e:
            self.single_result.config(text=f"出错: {e}", fg="#e74c3c")

    def start_folder_test(self):
        path = self.folder_path.get()
        if not path: messagebox.showwarning("提示", "请先选择数据集文件夹"); return
        threading.Thread(target=self._run_test, args=(path,), daemon=True).start()

    def _run_test(self, path):
        self.output.delete(1.0, "end")
        self.progress.pack(before=self.status); self.progress.start()
        self.status.config(text="加载中...")
        try:
            self.log(f"数据集: {path}", "info")
            X, y = [], []
            for d in range(10):
                sub = os.path.join(path, str(d))
                if not os.path.exists(sub): continue
                for fn in sorted(os.listdir(sub)):
                    if not fn.lower().endswith(('.png','.jpg','.jpeg','.bmp')): continue
                    gray = imread_auto(os.path.join(sub, fn))
                    if gray is None: continue
                    proc = preprocess_auto(gray)
                    if proc is None: continue
                    X.append(np.expand_dims(proc, -1)); y.append(d)
            if not X: self.log("错误: 未找到有效图像", "bad"); return
            total = len(X)
            self.log(f"加载: {total} 张", "info")

            self.status.config(text=f"预测中...")
            preds = predict_batch(X)
            y_pred = [p[0] for p in preds]; y_true = y
            acc = accuracy_score(y_true, y_pred)
            correct = int(acc * total)

            self.log("\n" + "="*50, "title")
            self.log(f"  测 试 结 果", "title")
            self.log("="*50, "title")
            self.log(f"  总样本: {total}    正确: {correct}    错误: {total-correct}", "info")
            self.log(f"  准确率: {acc*100:.2f}%", "good" if acc>0.9 else "bad")

            self.log(f"\n  各类别准确率:", "title")
            per_class = []
            for d in range(10):
                mask = np.array(y_true) == d
                total_d = mask.sum()
                if total_d == 0: continue
                correct_d = (np.array(y_pred)[mask] == d).sum()
                pct = correct_d/total_d*100
                bar = "#" * int(pct/2)
                self.log(f"    数字 {d}: {pct:5.1f}% ({correct_d}/{total_d}) {bar}",
                        "good" if pct>90 else ("bad" if pct<70 else None))
                per_class.append(pct)

            cm = confusion_matrix(y_true, y_pred)
            self.log(f"\n  混淆矩阵 (行=真实, 列=预测):", "title")
            present = [d for d in range(10) if (np.array(y_true)==d).sum()>0]
            self.log("       " + " ".join(f"{i:4d}" for i in present))
            for i in present:
                self.log(f"    {i}:  " + " ".join(f"{cm[i,j]:4d}" for j in present))

            self.log(f"\n  宏观平均准确率: {np.mean(per_class):.2f}%", "info")
            self.log(f"\n" + classification_report_str(y_true, y_pred), "info")
            self.status.config(text=f"完成 — 准确率: {acc*100:.2f}%")
        except Exception as e:
            import traceback
            self.log(f"\n错误: {e}", "bad")
            self.log(traceback.format_exc())
            self.status.config(text="出错")
        finally:
            self.progress.stop(); self.progress.pack_forget()


def main():
    root = tk.Tk()
    DigitTesterApp(root)
    root.mainloop()

if __name__ == '__main__':
    main()
