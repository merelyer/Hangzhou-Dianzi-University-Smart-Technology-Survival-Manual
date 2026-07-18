"""
混合训练：我的数据 + wx + aug500
训练: 40(我)+10(wx)+10(aug500) = 60/类 = 600
验证: 10(我) = 100 不变
测试: 各外部数据集剩余部分
"""

import os, cv2, json, numpy as np
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ============================================================
RANDOM_SEED = 42
OUTPUT_DIR = r"C:\Users\aa257\Desktop\复习\短学期作业\数据集\mixed_training"
os.makedirs(OUTPUT_DIR, exist_ok=True)

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

MY_DATA = r"C:\Users\aa257\Desktop\复习\短学期作业\数据集\原始增强分割图像"
WX_DATA = r"C:\Users\aa257\Desktop\复习\短学期作业\wx"
AUG500_DATA = r"C:\Users\aa257\Desktop\复习\短学期作业\dataset_aug500\dataset_aug500"
FRIEND_DATA = r"C:\Users\aa257\Desktop\复习\短学期作业\dataset\dataset"

TRAIN_MY = 40
TRAIN_EXT = 10  # per external dataset per class
VAL_MY = 10
BATCH = 16
LR = 0.002
EPOCHS = 80


def imread_u(path):
    with open(path, 'rb') as f:
        return cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_GRAYSCALE)


def imread_uc(path):
    with open(path, 'rb') as f:
        return cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)


def preprocess_my(raw_gray):
    """我的数据: 变尺寸灰度 → 28x28 黑底白字"""
    _, binary = cv2.threshold(raw_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(binary)
    if coords is None:
        return np.zeros((28,28), dtype=np.uint8)
    x, y, w, h = cv2.boundingRect(coords)
    pad = 3
    x1, y1 = max(0,x-pad), max(0,y-pad)
    x2, y2 = min(binary.shape[1],x+w+pad), min(binary.shape[0],y+h+pad)
    digit = binary[y1:y2, x1:x2]
    scale = 20.0 / max(digit.shape[0], digit.shape[1], 1)
    nh, nw = max(1,int(digit.shape[0]*scale)), max(1,int(digit.shape[1]*scale))
    resized = cv2.resize(digit, (nw,nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((28,28), dtype=np.uint8)
    off_y, off_x = (28-nh)//2, (28-nw)//2
    canvas[off_y:off_y+nh, off_x:off_x+nw] = resized
    return canvas


def preprocess_wx(raw_gray):
    """wx: 白底黑字 28x28 → 反转+Otsu"""
    # 统一28x28
    h, w = raw_gray.shape
    if h != 28 or w != 28:
        canvas = np.full((28,28), raw_gray[0,0].item(), dtype=np.uint8)
        off_y, off_x = max(0,(28-h)//2), max(0,(28-w)//2)
        crop_h, crop_w = min(h,28), min(w,28)
        canvas[off_y:off_y+crop_h, off_x:off_x+crop_w] = raw_gray[:crop_h,:crop_w]
        raw_gray = canvas
    inv = 255 - raw_gray
    _, binary = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def preprocess_friend(img_bgr):
    """朋友: 28x28 RGB 暗底亮字 → 阈值"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    bg = np.percentile(gray, 20)
    stretched = np.clip((gray.astype(np.float32)-bg)/(255-bg)*255, 0, 255).astype(np.uint8)
    _, binary = cv2.threshold(stretched, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(binary)
    if coords is None:
        return np.zeros((28,28), dtype=np.uint8)
    x, y, w, h = cv2.boundingRect(coords)
    pad = 2
    x1, y1 = max(0,x-pad), max(0,y-pad)
    x2, y2 = min(binary.shape[1],x+w+pad), min(binary.shape[0],y+h+pad)
    digit = binary[y1:y2, x1:x2]
    scale = 20.0 / max(digit.shape[0], digit.shape[1], 1)
    nh, nw = max(1,int(digit.shape[0]*scale)), max(1,int(digit.shape[1]*scale))
    resized = cv2.resize(digit, (nw,nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((28,28), dtype=np.uint8)
    off_y, off_x = (28-nh)//2, (28-nw)//2
    canvas[off_y:off_y+nh, off_x:off_x+nw] = resized
    return canvas


def load_dataset(path, preprocess_fn, mode='gray'):
    """加载一个数据集，按类返回图像列表"""
    data = {d: [] for d in range(10)}
    for d in range(10):
        folder = os.path.join(path, str(d))
        if not os.path.exists(folder):
            continue
        for f in sorted(os.listdir(folder)):
            if not f.endswith(('.png','.jpg','.jpeg')):
                continue
            if mode == 'color':
                img = imread_uc(os.path.join(folder, f))
            else:
                img = imread_u(os.path.join(folder, f))
            if img is None:
                continue
            processed = preprocess_fn(img)
            data[d].append(processed.astype(np.float32)/255.0)
    return data


# ============================================================
print("=" * 60)
print("加载各数据集...")

my_data = load_dataset(MY_DATA, preprocess_my, 'gray')
wx_data = load_dataset(WX_DATA, preprocess_wx, 'gray')
aug500_data = load_dataset(AUG500_DATA, lambda x: x, 'gray')  # 已是28x28黑底白字
friend_data = load_dataset(FRIEND_DATA, preprocess_friend, 'color')

for name, d in [('My', my_data), ('wx', wx_data), ('aug500', aug500_data), ('Friend', friend_data)]:
    counts = [len(d[i]) for i in range(10)]
    print('  %s: %s (total=%d)' % (name, str(counts), sum(counts)))

# ============================================================
# 构建混合训练集
print("\n" + "=" * 60)
print("构建混合训练集...")

X_train, y_train = [], []
X_val, y_val = [], []

for d in range(10):
    # 训练: 我的 + wx + aug500
    np.random.shuffle(my_data[d])
    np.random.shuffle(wx_data[d])
    np.random.shuffle(aug500_data[d])

    X_train.extend(my_data[d][:TRAIN_MY])
    X_train.extend(wx_data[d][:TRAIN_EXT])
    X_train.extend(aug500_data[d][:TRAIN_EXT])
    y_train.extend([d] * (TRAIN_MY + TRAIN_EXT + TRAIN_EXT))

    # 验证: 仅我的数据
    X_val.extend(my_data[d][TRAIN_MY:TRAIN_MY + VAL_MY])
    y_val.extend([d] * VAL_MY)

X_train = np.expand_dims(np.array(X_train), -1)
y_train_cat = keras.utils.to_categorical(np.array(y_train), 10)
X_val = np.expand_dims(np.array(X_val), -1)
y_val = np.array(y_val)
y_val_cat = keras.utils.to_categorical(y_val, 10)

print("Train: %d (%d/class)  Val: %d (%d/class)" % (
    len(X_train), len(X_train)//10, len(X_val), len(X_val)//10))

# ============================================================
# 构建模型 & 训练
def build_model():
    m = keras.Sequential(name='MixedCNN')
    m.add(layers.Conv2D(32,(3,3),padding='same',activation='relu',
          kernel_regularizer=keras.regularizers.l2(1e-4),input_shape=(28,28,1)))
    m.add(layers.MaxPooling2D((2,2))); m.add(layers.Dropout(0.25))
    m.add(layers.Conv2D(64,(3,3),padding='same',activation='relu',
          kernel_regularizer=keras.regularizers.l2(1e-4)))
    m.add(layers.MaxPooling2D((2,2))); m.add(layers.Dropout(0.3))
    m.add(layers.Flatten())
    m.add(layers.Dense(128,activation='relu',kernel_regularizer=keras.regularizers.l2(1e-4)))
    m.add(layers.Dropout(0.5))
    m.add(layers.Dense(10,activation='softmax'))
    return m

model = build_model()
model.compile(optimizer=keras.optimizers.Adam(LR),
              loss='categorical_crossentropy', metrics=['accuracy'])

callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True, verbose=1),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6, verbose=1),
    keras.callbacks.ModelCheckpoint(os.path.join(OUTPUT_DIR, 'best_model.keras'),
                                     monitor='val_accuracy', save_best_only=True, verbose=1),
]

augmenter = keras.Sequential([
    layers.RandomRotation(0.04, fill_mode='constant', fill_value=0),
    layers.RandomTranslation(0.12, 0.12, fill_mode='constant', fill_value=0),
    layers.RandomZoom(0.15, 0.15, fill_mode='constant', fill_value=0),
    layers.RandomContrast(0.25),
])

train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train_cat))
train_ds = train_ds.shuffle(1024).batch(BATCH)
train_ds = train_ds.map(lambda x, y: (augmenter(x, training=True), y),
                        num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val_cat))
val_ds = val_ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)

print("\n" + "=" * 60)
print("训练混合模型...")
model.fit(train_ds, epochs=EPOCHS, validation_data=val_ds, callbacks=callbacks, verbose=2)

# ============================================================
# 各数据集测试
def test_on_dataset(name, X_test, y_test):
    y_pred = np.argmax(model.predict(X_test, batch_size=64, verbose=0), axis=1)
    acc = accuracy_score(y_test, y_pred)
    per_class = confusion_matrix(y_test, y_pred).diagonal() / np.maximum(
        np.bincount(y_test, minlength=10), 1)
    return acc, per_class

# 构建各测试集（训练后剩余部分）
results = {}
for ds_name, ds_data in [('wx', wx_data), ('aug500', aug500_data), ('Friend', friend_data)]:
    X_test, y_test = [], []
    for d in range(10):
        start = TRAIN_EXT  # 前10张用于训练，剩余用于测试
        X_test.extend(ds_data[d][start:])
        y_test.extend([d] * len(ds_data[d][start:]))
    if X_test:
        X_test = np.expand_dims(np.array(X_test), -1)
        y_test = np.array(y_test)
        acc, per_class = test_on_dataset(ds_name, X_test, y_test)
        results[ds_name] = (acc, per_class, len(y_test))
        print('\n--- %s (%d samples) ---' % (ds_name, len(y_test)))
        print('Accuracy: %.4f (%.2f%%)' % (acc, acc*100))
        for i, p in enumerate(per_class):
            bar = '#' * int(p*50)
            print('  %d: %.4f (%5.1f%%) %s' % (i, p, p*100, bar))

# 保存
model.save(os.path.join(OUTPUT_DIR, 'final_model.keras'))
with open(os.path.join(OUTPUT_DIR, 'results.json'), 'w', encoding='utf-8') as f:
    json.dump({k: {'acc': float(v[0]), 'per_class': [float(x) for x in v[1]], 'n': v[2]}
               for k, v in results.items()}, f, indent=2)

print("\n" + "=" * 60)
print("混合训练完成! 模型: %s" % OUTPUT_DIR)
