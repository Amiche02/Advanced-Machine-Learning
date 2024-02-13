import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from datetime import datetime
import matplotlib.pyplot as plt

num_classes = 10

#load data
mnist = keras.datasets.mnist

#split between train and validation sets
(x_train, y_train), (x_val, y_val) = mnist.load_data()

#convert images into one dimension from 28x28 pixels
x_train = x_train.reshape(x_train.shape[0], -1)
x_val = x_val.reshape(x_val.shape[0], -1)
x_train = x_train.astype('float32')
x_val = x_val.astype('float32')

#convert labels to one-hot vectors
y_train = to_categorical(y_train, num_classes=num_classes)
y_val = to_categorical(y_val, num_classes=num_classes)

model = keras.Sequential([
    keras.layers.Dense(512, input_shape=(28*28,)),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(num_classes, activation='softmax')
])

model.summary()

#compile the model
model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=["accuracy"])

logdir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
checkpointer = ModelCheckpoint(filepath="model.hdf5", monitor="val_loss", verbose=1, save_best_only=True)
#train model
history = model.fit(x_train, y_train,
                    validation_data=(x_val, y_val),
                    epochs=10,
                    callbacks=[checkpointer, tensorboard_callback])

score = model.evaluate(x_val, y_val, verbose=0)

print("\nScore : ",score)
print("Test loss:", score[0])
print("Test accuracy:", score[1])


#ploting
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['vazl_accuracy'], label='validation')
plt.title("Model accuracy")
plt.xlabel("accuracy")
plt.ylabel('epoch')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='validation')
plt.title("Model accuracy")
plt.xlabel("accuracy")
plt.ylabel('epoch')
plt.legend()

plt.tight_layout()
plt.show()




