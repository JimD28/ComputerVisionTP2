# This code is adapted from:
# https://github.com/priya-dwivedi/Deep-Learning/blob/master/resnet_keras/Residual_Network_Keras.ipynb

import numpy as np
import matplotlib.pyplot as plt
import os
from urllib.request import urlopen, urlretrieve
from PIL import Image
from sklearn.utils import shuffle
import cv2
from resnets_utils import *
import pickle

from tensorflow.keras.models import load_model
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from glob import glob
from tensorflow.keras import applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, GlobalAveragePooling2D
from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau, ModelCheckpoint

with open('dataset_color.pkl', 'rb') as g:  # Python 3: open(..., 'rb')
    ellipses, labels = pickle.load(g)

X_train, X_test, y_train, y_test = train_test_split(ellipses, labels, test_size=0.3, random_state=10000)
with open('datasets_3.pkl', 'wb') as f:
    pickle.dump([X_train, X_test, y_train, y_test], f)

checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = ModelCheckpoint(filepath=checkpoint_path,
                              save_weights_only=True,
                              verbose=1)

print("number of training examples = " + str(X_train.shape[0]))
print("number of test examples = " + str(X_test.shape[0]))
print("X_train shape: " + str(X_train.shape))
print("Y_train shape: " + str(y_train.shape))
print("X_test shape: " + str(X_test.shape))
print("Y_test shape: " + str(y_test.shape))

img_height, img_width = 240, 320
num_classes = 4
# If imagenet weights are being loaded,
# input must have a static square shape (one of (128, 128), (160, 160), (192, 192), or (224, 224))
base_model = applications.resnet50.ResNet50(weights=None, include_top=False, input_shape=(img_height, img_width, 3))

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.7)(x)
predictions = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

from tensorflow.keras.optimizers import SGD, Adam

# sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
adam = Adam(lr=0.0001)
model.compile(optimizer=adam, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=100, batch_size=16, callbacks=[cp_callback])

preds = model.evaluate(X_test, y_test)
print("Loss = " + str(preds[0]))
print("Test Accuracy = " + str(preds[1]))

model.summary()

model.save_weights('./checkpoints/my_checkpoint')
model.save('my_model_3.h5')

# new_model = tf.keras.models.load_model('my_model.h5')
