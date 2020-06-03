from sklearn.model_selection import train_test_split
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from os import listdir
import numpy as np
from mtcnn.mtcnn import MTCNN



label_names = {}
forder = "dataset/raw"
index_label = 0
for forder_name in listdir(forder):
    label_names[forder_name] = index_label 
    index_label += 1
print(label_names)       
# {'van_long': 0, 'tran_thanh': 1, 'Bich_lan': 2, 'hoai_linh': 3}

y_labels = []
x_data = []
for forder_name in listdir(forder):
    for filename in listdir(forder + "/" + forder_name):
        y_labels.append(label_names[forder_name])
        path = forder + "/" + forder_name + "/" + filename
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.cast(image, tf.float32)
        image = (image/255)           #normalize
        image = tf.image.resize(image, (256, 256))
        x_data.append(image)
print("")
print(set(y_labels))

# print(y_labels)
x_data = tf.convert_to_tensor(x_data)
# print(x_data.shape)
# print(type(x_data[0]))
y_labels = tf.convert_to_tensor(y_labels)
# x_data = np.array(x_data)
# y_labels = np.array(y_labels)

print(type(x_data))
print("=======")
print("")

print("")
X_train, X_test, y_train, y_test = train_test_split(x_data.numpy(), y_labels.numpy(), test_size=0.2)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
print("")

# print(X_train.shape)
# print(X_test.shape)
# print(y_train.shape)
# print(y_test.shape)
# print(type(X_train))
# print(X_train[:3])
X_train = tf.convert_to_tensor(X_train)
X_test =    tf.convert_to_tensor(X_test)
y_train = tf.convert_to_tensor(y_train)
y_test = tf.convert_to_tensor(y_test)

# print(y_test[:20])


model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(20, 3, activation='relu'),

    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(20, 5, activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),

    # TODO: fill suitable activations
    tf.keras.layers.Dense(units=40, activation='relu'),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(units=30, activation='relu'),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(units=4, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy',
            optimizer = tf.keras.optimizers.Adam(),
            metrics = ['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size = 10)
# serialize weights to HDF5
model.save_weights("model_weights1.h5")
model.save("model1.h5")
print(model.evaluate(X_test,y_test))
# model.load_weights("model.h5")
