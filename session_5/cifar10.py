import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import Model, layers, callbacks
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.utils import plot_model

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")

def dislay_image(x, y, nrows=4, ncols=6):
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    for i in range(nrows*ncols):
        idx = np.random.randint(0, x.shape[0])
        plt.subplot(nrows, ncols, i+1)
        plt.imshow(x[idx])
        plt.title(f"{class_names[np.argmax(y[idx])]}")
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)


    plt.tight_layout()
    plt.show()


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
assert x_train.shape == (50000, 32, 32, 3)
assert x_test.shape == (10000, 32, 32, 3)
assert y_train.shape == (50000, 1)
assert y_test.shape == (10000, 1)

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, random_state=42, test_size=0.2, stratify=y_train, shuffle=True)


y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)
y_val = to_categorical(y_val, num_classes=10)

print(f"x_train = {x_train.shape}   y_train = {y_train.shape}\nx_test = {x_test.shape}   y_test = {y_test.shape}\
\nx_val = {x_val.shape}   y_val = {y_val.shape}")

dislay_image(x_train, y_train)

x_train = preprocess_input(x_train)/255.0
x_test = preprocess_input(x_test)/255.0
x_val = preprocess_input(x_val)/255.0

# Model
num_classes = y_train.shape[1]
base_model = VGG16(weights="imagenet", include_top=False, input_shape=(32, 32, 3))

base_output = base_model.output
x =  layers.Flatten()(base_output)
x = layers.Dense(3000, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(1500, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)
preds = layers.Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=preds)

for layer in model.layers[:19]:
    layer.trainable = False
for layer in model.layers[19:]:
    layer.trainable = True


model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              metrics=["accuracy"])


tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir = "tb_callback_dir", histogram_freq=1,
)

history = model.fit(x_train, y_train,
    epochs=10,
    validation_batch_size=32,
    validation_data=(x_val, y_val),
    callbacks=[tensorboard_callback])