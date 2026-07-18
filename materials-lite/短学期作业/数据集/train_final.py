"""
最终混合训练: 加入 aug_28x28 样本
训练后自动转换 ONNX + 复制到 dist
"""

import os, cv2, shutil, numpy as np, subprocess

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

np.random.seed(42); tf.random.set_seed(42)

BASE = r'C:\Users\aa257\Desktop\复习\短学期作业'
OUT_DIR = os.path.join(BASE, '数据集', 'final_model')
os.makedirs(OUT_DIR, exist_ok=True)

# =========== 预处理 ===========
def imread_u(path):
    with open(path, 'rb') as f:
        return cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_GRAYSCALE)

def imread_uc(path):
    with open(path, 'rb') as f:
        return cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)

def preproc_my(g):
    _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    c = cv2.findNonZero(b)
    if c is None: return np.zeros((28, 28), dtype=np.uint8)
    x, y, w, h = cv2.boundingRect(c); p = 3
    x1, y1 = max(0, x-p), max(0, y-p); x2, y2 = min(b.shape[1], x+w+p), min(b.shape[0], y+h+p)
    d = b[y1:y2, x1:x2]; s = 20.0 / max(d.shape[0], d.shape[1], 1)
    nh, nw = max(1, int(d.shape[0]*s)), max(1, int(d.shape[1]*s))
    r = cv2.resize(d, (nw, nh), interpolation=cv2.INTER_AREA)
    canv = np.zeros((28, 28), dtype=np.uint8); oy, ox = (28-nh)//2, (28-nw)//2
    canv[oy:oy+nh, ox:ox+nw] = r; return canv

def preproc_wx(g):
    h, w = g.shape
    if h != 28 or w != 28:
        canv = np.full((28, 28), g[0, 0].item(), dtype=np.uint8)
        oy, ox = max(0, (28-h)//2), max(0, (28-w)//2)
        ch, cw = min(h, 28), min(w, 28); canv[oy:oy+ch, ox:ox+cw] = g[:ch, :cw]; g = canv
    inv = 255 - g; _, b = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return b

def preproc_friend(ic):
    g = cv2.cvtColor(ic, cv2.COLOR_BGR2GRAY)
    bg = np.percentile(g, 20)
    if bg >= 255: bg = 50
    st = np.clip((g.astype(np.float32)-bg)/(255-bg)*255, 0, 255).astype(np.uint8)
    _, b = cv2.threshold(st, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    c = cv2.findNonZero(b)
    if c is None: return np.zeros((28, 28), dtype=np.uint8)
    x, y, w, h = cv2.boundingRect(c); p = 2
    x1, y1 = max(0, x-p), max(0, y-p); x2, y2 = min(b.shape[1], x+w+p), min(b.shape[0], y+h+p)
    d = b[y1:y2, x1:x2]; s = 20.0 / max(d.shape[0], d.shape[1], 1)
    nh, nw = max(1, int(d.shape[0]*s)), max(1, int(d.shape[1]*s))
    r = cv2.resize(d, (nw, nh), interpolation=cv2.INTER_AREA)
    canv = np.zeros((28, 28), dtype=np.uint8); oy, ox = (28-nh)//2, (28-nw)//2
    canv[oy:oy+nh, ox:ox+nw] = r; return canv

# aug_28x28 预处理 — JPG 压缩图，Otsu 阈值后居中
def preproc_aug28(g):
    # JPG 压缩导致值在 [50,170] 范围，需 Otsu
    _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    c = cv2.findNonZero(b)
    if c is None: return np.zeros((28, 28), dtype=np.uint8)
    x, y, w, h = cv2.boundingRect(c); p = 3
    x1, y1 = max(0, x-p), max(0, y-p); x2, y2 = min(b.shape[1], x+w+p), min(b.shape[0], y+h+p)
    d = b[y1:y2, x1:x2]; s = 20.0 / max(d.shape[0], d.shape[1], 1)
    nh, nw = max(1, int(d.shape[0]*s)), max(1, int(d.shape[1]*s))
    r = cv2.resize(d, (nw, nh), interpolation=cv2.INTER_AREA)
    canv = np.zeros((28, 28), dtype=np.uint8); oy, ox = (28-nh)//2, (28-nw)//2
    canv[oy:oy+nh, ox:ox+nw] = r; return canv


# =========== 加载所有数据集 ===========
def load_ds(path, fn, mode='gray'):
    data = {d: [] for d in range(10)}
    for d in range(10):
        fld = os.path.join(path, str(d))
        if not os.path.exists(fld): continue
        for f in sorted(os.listdir(fld)):
            if not f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')): continue
            try:
                if mode == 'color':
                    img = imread_uc(os.path.join(fld, f))
                else:
                    img = imread_u(os.path.join(fld, f))
                if img is None: continue
                data[d].append(fn(img).astype(np.float32) / 255.0)
            except Exception:
                continue
    return data

print("Loading datasets...")
my  = load_ds(os.path.join(BASE, '数据集/原始增强分割图像'), preproc_my)
wx  = load_ds(os.path.join(BASE, 'wx'), preproc_wx)
a5  = load_ds(os.path.join(BASE, 'dataset_aug500/dataset_aug500'), lambda x: x)
fr  = load_ds(os.path.join(BASE, 'dataset/dataset'), preproc_friend, 'color')
a28 = load_ds(os.path.join(BASE, 'aug_28x28/aug_28x28'), preproc_aug28)

for name, ds in [('My', my), ('wx', wx), ('aug500', a5), ('Friend', fr), ('aug28', a28)]:
    print(f'  {name}: {sum(len(v) for v in ds.values())}')

# =========== 构建训练集 ===========
X_tr, y_tr = [], []
T_MY, T_WX, T_A5, T_A28 = 40, 15, 15, 15

for d in range(10):
    for ds, n in [(my, T_MY), (wx, T_WX), (a5, T_A5), (fr, len(fr[d])), (a28, T_A28)]:
        avail = ds.get(d, [])
        np.random.shuffle(avail)
        take = min(n, len(avail))
        X_tr.extend(avail[:take])
        y_tr.extend([d] * take)

X_tr = np.expand_dims(np.array(X_tr), -1)
y_tr_c = keras.utils.to_categorical(np.array(y_tr), 10)
print(f'\nTrain: {len(X_tr)} ({len(X_tr)//10}/class)')

# 验证集 — 仅我的数据
X_vl, y_vl = [], []
for d in range(10):
    X_vl.extend(my[d][T_MY:T_MY+10])
    y_vl.extend([d] * 10)
X_vl = np.expand_dims(np.array(X_vl), -1)
y_vl_c = keras.utils.to_categorical(np.array(y_vl), 10)

# =========== 训练 ===========
model = keras.Sequential(name='FinalCNN')
model.add(layers.Conv2D(32, (3,3), padding='same', activation='relu',
          kernel_regularizer=keras.regularizers.l2(1e-4), input_shape=(28,28,1)))
model.add(layers.MaxPooling2D((2,2))); model.add(layers.Dropout(0.25))
model.add(layers.Conv2D(64, (3,3), padding='same', activation='relu',
          kernel_regularizer=keras.regularizers.l2(1e-4)))
model.add(layers.MaxPooling2D((2,2))); model.add(layers.Dropout(0.3))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-4)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer=keras.optimizers.Adam(0.002),
              loss='categorical_crossentropy', metrics=['accuracy'])

augmenter = keras.Sequential([
    layers.RandomRotation(0.05, fill_mode='constant', fill_value=0),
    layers.RandomTranslation(0.14, 0.14, fill_mode='constant', fill_value=0),
    layers.RandomZoom(0.16, 0.16, fill_mode='constant', fill_value=0),
    layers.RandomContrast(0.3),
])

tr_ds = tf.data.Dataset.from_tensor_slices((X_tr, y_tr_c)).shuffle(2048).batch(16)
tr_ds = tr_ds.map(lambda x, y: (augmenter(x, training=True), y),
                  num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
vl_ds = tf.data.Dataset.from_tensor_slices((X_vl, y_vl_c)).batch(16).prefetch(tf.data.AUTOTUNE)

cb = [keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=25, restore_best_weights=True, verbose=1),
      keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, verbose=1)]

print("\nTraining...")
model.fit(tr_ds, epochs=100, validation_data=vl_ds, callbacks=cb, verbose=2)

# 保存 Keras 模型
keras_path = os.path.join(OUT_DIR, 'best_model.keras')
model.save(keras_path)
print(f'Model saved: {keras_path}')

# =========== 转换 ONNX ===========
print("\nConverting to ONNX...")
tmp_saved = r'C:\Users\aa257\tmp_saved_model'
if os.path.exists(tmp_saved): shutil.rmtree(tmp_saved)
tf.saved_model.save(model, tmp_saved)

onnx_path = os.path.join(OUT_DIR, 'model.onnx')
subprocess.run(['python', '-m', 'tf2onnx.convert', '--saved-model', tmp_saved,
                '--output', onnx_path, '--opset', '13'], check=True, capture_output=True)
shutil.rmtree(tmp_saved)
print(f'ONNX saved: {onnx_path} ({os.path.getsize(onnx_path)//1024} KB)')

# 复制到 dist
dist_path = os.path.join(BASE, '数据集', 'dist', 'DigitTester', '_internal', 'model.onnx')
shutil.copy2(onnx_path, dist_path)
print(f'ONNX copied to dist: {dist_path}')

# 也复制到数据集根目录给 build_exe 用
shutil.copy2(onnx_path, os.path.join(BASE, '数据集', 'model.onnx'))
print('ONNX also copied to dataset root for build_exe')

# =========== 快速测试 ===========
import onnxruntime as ort
sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
inp_name = sess.get_inputs()[0].name

# 测试 aug_28x28
X_test, y_test = [], []
for d in range(10):
    X_test.extend(a28[d][T_A28:])  # 训练用剩的
    y_test.extend([d] * len(a28[d][T_A28:]))
if X_test and len(X_test) > 100:
    X_test = np.expand_dims(np.array(X_test), -1)
    y_test = np.array(y_test)
    yp = []
    for x in X_test:
        out = sess.run(None, {inp_name: np.expand_dims(x, 0)})[0]
        yp.append(np.argmax(out))
    acc = (y_test == np.array(yp)).mean()
    print(f'\nQuick test on aug_28x28 holdout: {len(X_test)} samples, accuracy={acc*100:.2f}%')

print('\nDone! Now run: python build_exe.py')
