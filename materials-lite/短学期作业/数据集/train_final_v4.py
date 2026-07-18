"""
最终训练 V4: 加入 MNIST 提升泛化能力
MNIST (60000 张, 500+ 不同人手写) → 基础识别能力
用户数据集 → 保持对特定风格的高精度
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

# =========== 预处理函数 ===========
def imread_u(path):
    with open(path, 'rb') as f:
        return cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_GRAYSCALE)

def imread_uc(path):
    with open(path, 'rb') as f:
        return cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)

def center_digit(binary, pad=3):
    c = cv2.findNonZero(binary)
    if c is None: return np.zeros((28, 28), dtype=np.uint8)
    x, y, w, h = cv2.boundingRect(c)
    x1, y1 = max(0, x-pad), max(0, y-pad)
    x2, y2 = min(binary.shape[1], x+w+pad), min(binary.shape[0], y+h+pad)
    d = binary[y1:y2, x1:x2]
    if d.size == 0: return np.zeros((28, 28), dtype=np.uint8)
    s = 20.0 / max(d.shape[0], d.shape[1], 1)
    nh, nw = max(1, int(d.shape[0]*s)), max(1, int(d.shape[1]*s))
    r = cv2.resize(d, (nw, nh), interpolation=cv2.INTER_AREA)
    canv = np.zeros((28, 28), dtype=np.uint8)
    oy, ox = (28-nh)//2, (28-nw)//2
    canv[oy:oy+nh, ox:ox+nw] = r
    return canv

def preproc_my(g):
    _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return center_digit(b)

def preproc_wx(g):
    h, w = g.shape
    if h != 28 or w != 28:
        canv = np.full((28, 28), g[0, 0].item(), dtype=np.uint8)
        oy, ox = max(0, (28-h)//2), max(0, (28-w)//2)
        ch, cw = min(h, 28), min(w, 28)
        canv[oy:oy+ch, ox:ox+cw] = g[:ch, :cw]
        g = canv
    inv = 255 - g
    _, b = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return b

def preproc_friend(ic):
    g = cv2.cvtColor(ic, cv2.COLOR_BGR2GRAY)
    bg = np.percentile(g, 20)
    if bg >= 255: bg = 50
    st = np.clip((g.astype(np.float32)-bg)/(255-bg)*255, 0, 255).astype(np.uint8)
    _, b = cv2.threshold(st, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return center_digit(b, pad=2)

def preproc_aug28(g):
    edges = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    edge_mean = float(np.mean(edges))
    if edge_mean > 128:
        _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return center_digit(b)

def preproc_augimg(g):
    inv = 255 - g
    _, b = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return center_digit(b)

# =========== 加载 MNIST ===========
print("Loading MNIST...")
(x_mnist, y_mnist), (x_mnist_test, y_mnist_test) = keras.datasets.mnist.load_data()
# MNIST 已经是 28x28 黑底白字，直接归一化
x_mnist = x_mnist.astype(np.float32) / 255.0
x_mnist_test = x_mnist_test.astype(np.float32) / 255.0
print(f'  MNIST train: {len(x_mnist)}, test: {len(x_mnist_test)}')

# 用一部分 MNIST（15000张，每类1500张）混合用户数据
mnist_idx = []
for d in range(10):
    d_idx = np.where(y_mnist == d)[0]
    np.random.shuffle(d_idx)
    mnist_idx.extend(d_idx[:1500])  # 每类3000张
np.random.shuffle(mnist_idx)
X_mnist = x_mnist[mnist_idx]
y_mnist = y_mnist[mnist_idx]
print(f'  Using {len(X_mnist)} MNIST samples for training (3000/class)')

# =========== 加载用户数据集 ===========
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

print("\nLoading user datasets...")
my  = load_ds(os.path.join(BASE, '数据集/原始增强分割图像'), preproc_my)
wx  = load_ds(os.path.join(BASE, 'wx'), preproc_wx)
a5  = load_ds(os.path.join(BASE, 'dataset_aug500/dataset_aug500'), lambda x: x)
fr  = load_ds(os.path.join(BASE, 'dataset/dataset'), preproc_friend, 'color')
a28 = load_ds(os.path.join(BASE, 'aug_28x28/aug_28x28'), preproc_aug28)
aimg = load_ds(os.path.join(BASE, 'augmented_images/augmented_images'), preproc_augimg)

for name, ds in [('My', my), ('wx', wx), ('aug500', a5), ('Friend', fr),
                  ('aug28', a28), ('augImg', aimg)]:
    print(f'  {name}: {sum(len(v) for v in ds.values())} images')

# =========== 构建混合训练集 ===========
X_tr, y_tr = [], []

# 先用用户数据
T_MY, T_WX, T_A5, T_A28, T_AIMG = 40, 15, 15, 150, 50
for d in range(10):
    for ds, n in [(my, T_MY), (wx, T_WX), (a5, T_A5), (fr, len(fr[d])),
                   (a28, T_A28), (aimg, T_AIMG)]:
        avail = ds.get(d, [])
        np.random.shuffle(avail)
        take = min(n, len(avail))
        X_tr.extend(avail[:take])
        y_tr.extend([d] * take)

# 然后添加 MNIST（每类3000张）
for i in range(len(X_mnist)):
    X_tr.append(X_mnist[i])
    y_tr.append(int(y_mnist[i]))

X_tr = np.expand_dims(np.array(X_tr, dtype=np.float32), -1)
y_tr = np.array(y_tr)
y_tr_c = keras.utils.to_categorical(y_tr, 10)
print(f'\nTotal train: {len(X_tr)} images')
print(f'  User data: {len(X_tr) - len(X_mnist)} + MNIST: {len(X_mnist)}')
print(f'  Distribution: {np.bincount(y_tr)}')

# 验证集: 用户数据 holdout + MNIST test
X_vl, y_vl = [], []
for d in range(10):
    X_vl.extend(my[d][T_MY:T_MY+10])
    y_vl.extend([d] * 10)
# 加一些 MNIST test
mnist_test_idx = []
for d in range(10):
    d_idx = np.where(y_mnist_test == d)[0][:50]
    mnist_test_idx.extend(d_idx)
X_vl.extend(x_mnist_test[mnist_test_idx])
y_vl.extend(y_mnist_test[mnist_test_idx])

X_vl = np.expand_dims(np.array(X_vl, dtype=np.float32), -1)
y_vl_c = keras.utils.to_categorical(np.array(y_vl), 10)
print(f'Validation: {len(X_vl)} images')

# =========== 更大的模型 ===========
model = keras.Sequential(name='GeneralCNN')
model.add(layers.Conv2D(32, (3,3), padding='same', activation='relu',
          kernel_regularizer=keras.regularizers.l2(1e-4), input_shape=(28,28,1)))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(32, (3,3), padding='same', activation='relu',
          kernel_regularizer=keras.regularizers.l2(1e-4)))
model.add(layers.MaxPooling2D((2,2))); model.add(layers.Dropout(0.25))
model.add(layers.Conv2D(64, (3,3), padding='same', activation='relu',
          kernel_regularizer=keras.regularizers.l2(1e-4)))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(64, (3,3), padding='same', activation='relu',
          kernel_regularizer=keras.regularizers.l2(1e-4)))
model.add(layers.MaxPooling2D((2,2))); model.add(layers.Dropout(0.35))
model.add(layers.Conv2D(128, (3,3), padding='same', activation='relu',
          kernel_regularizer=keras.regularizers.l2(1e-4)))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2,2))); model.add(layers.Dropout(0.35))
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-4)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer=keras.optimizers.Adam(0.001),
              loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# 数据增强
augmenter = keras.Sequential([
    layers.RandomRotation(0.1, fill_mode='constant', fill_value=0),
    layers.RandomTranslation(0.15, 0.15, fill_mode='constant', fill_value=0),
    layers.RandomZoom(0.2, 0.2, fill_mode='constant', fill_value=0),
    layers.RandomContrast(0.3),
])

tr_ds = tf.data.Dataset.from_tensor_slices((X_tr, y_tr_c)).shuffle(65536).batch(64)
tr_ds = tr_ds.map(lambda x, y: (augmenter(x, training=True), y),
                  num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
vl_ds = tf.data.Dataset.from_tensor_slices((X_vl, y_vl_c)).batch(64).prefetch(tf.data.AUTOTUNE)

cb = [keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True, verbose=1),
      keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6, verbose=1)]

print("\nTraining...")
history = model.fit(tr_ds, epochs=100, validation_data=vl_ds, callbacks=cb, verbose=2)

best_val_acc = max(history.history['val_accuracy'])
print(f'\nBest val_accuracy: {best_val_acc*100:.2f}%')

# 保存
keras_path = os.path.join(OUT_DIR, 'best_model.keras')
model.save(keras_path)
print(f'Keras saved: {keras_path}')

# ONNX
print("\nConverting to ONNX...")
tmp_saved = r'C:\Users\aa257\tmp_saved_model'
if os.path.exists(tmp_saved): shutil.rmtree(tmp_saved)
tf.saved_model.save(model, tmp_saved)

onnx_path = os.path.join(OUT_DIR, 'model.onnx')
subprocess.run(['python', '-m', 'tf2onnx.convert', '--saved-model', tmp_saved,
                '--output', onnx_path, '--opset', '13'], check=True, capture_output=True)
shutil.rmtree(tmp_saved)
print(f'ONNX: {onnx_path} ({os.path.getsize(onnx_path)//1024} KB)')

# 复制
dist_path = os.path.join(BASE, '数据集', 'dist', 'DigitTester', '_internal', 'model.onnx')
shutil.copy2(onnx_path, dist_path)
shutil.copy2(onnx_path, os.path.join(BASE, '数据集', 'model.onnx'))
print('ONNX copied to dist and root')

# =========== 快速测试 ===========
import onnxruntime as ort
sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
inp_name = sess.get_inputs()[0].name

print('\n' + '='*60)
print('  Quick Tests')
print('='*60)

# MNIST test
X_mt = np.expand_dims(x_mnist_test[:2000], -1)
yp_mt = []
for x in X_mt:
    out = sess.run(None, {inp_name: np.expand_dims(x, 0)})[0]
    yp_mt.append(np.argmax(out))
acc_mt = (y_mnist_test[:2000] == np.array(yp_mt)).mean()
print(f'MNIST test (2000): accuracy={acc_mt*100:.2f}%')

# aug_28x28 holdout
X_t28, y_t28 = [], []
for d in range(10):
    holdout = a28[d][150:]
    X_t28.extend(holdout)
    y_t28.extend([d] * len(holdout))
if X_t28:
    X_t28 = np.expand_dims(np.array(X_t28), -1)
    yp28 = []
    for x in X_t28:
        out = sess.run(None, {inp_name: np.expand_dims(x, 0)})[0]
        yp28.append(np.argmax(out))
    acc28 = (np.array(y_t28) == np.array(yp28)).mean()
    print(f'aug_28x28 holdout: {len(X_t28)} samples, accuracy={acc28*100:.2f}%')

# augmented_images holdout
X_tai, y_tai = [], []
for d in range(10):
    holdout = aimg[d][50:]
    X_tai.extend(holdout)
    y_tai.extend([d] * len(holdout))
if X_tai:
    X_tai = np.expand_dims(np.array(X_tai), -1)
    ypai = []
    for x in X_tai:
        out = sess.run(None, {inp_name: np.expand_dims(x, 0)})[0]
        ypai.append(np.argmax(out))
    accai = (np.array(y_tai) == np.array(ypai)).mean()
    print(f'augmented_images holdout: {len(X_tai)} samples, accuracy={accai*100:.2f}%')

print('\nDone! New model has MNIST generalization + user data specialization.')
