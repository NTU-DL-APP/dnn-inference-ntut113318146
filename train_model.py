import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout
from tensorflow.keras import regularizers
from utils import mnist_reader

# === 1. 載入資料（目前用 t10k 當訓練資料）===
x_train, y_train = mnist_reader.load_mnist('data/fashion', kind='t10k')
x_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')

x_train = x_train / 255.0
x_test = x_test / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# === 2. 建立模型 ===
model = Sequential([
    Dense(256, activation='relu', input_shape=(784,)),
    Dropout(0.3),  # ← 新增
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# === 3. EarlyStopping 回呼函式 ===
early_stop = EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True
)

# === 4. 訓練模型 ===
history = model.fit(
    x_train, y_train,
    epochs=50,
    batch_size=128,
    validation_split=0.2,
    callbacks=[early_stop]
)

# === 5. 評估模型 ===
loss, acc = model.evaluate(x_test, y_test)
print("Test Accuracy:", acc)

# === 6. 儲存模型 ===
model.save("fashion_model.h5")
print("✅ 模型已儲存為 fashion_model.h5")

# === 7. 匯出 JSON + NPZ 給 numpy 推論用 ===
model_arch = []
weights_dict = {}

for i, layer in enumerate(model.layers):
    if isinstance(layer, tf.keras.layers.Dense):
        act = layer.activation.__name__
        W, b = layer.get_weights()
        W_name = f"W_{i}"
        b_name = f"b_{i}"
        model_arch.append({
            "name": f"layer_{i}",
            "type": "Dense",
            "config": {"activation": act},
            "weights": [W_name, b_name]
        })
        weights_dict[W_name] = W
        weights_dict[b_name] = b

os.makedirs("model", exist_ok=True)

with open("model/fashion_mnist.json", "w") as f:
    json.dump(model_arch, f, indent=2)
np.savez("model/fashion_mnist.npz", **weights_dict)
print("✅ 已輸出 model/fashion_mnist.json 與 model/fashion_mnist.npz")

# === 8. 繪圖：acc/loss 曲線 ===
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig('accuracy_curve.png')
plt.close()

plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('loss_curve.png')
plt.close()

print("✅ 已繪製 accuracy_curve.png 與 loss_curve.png")
