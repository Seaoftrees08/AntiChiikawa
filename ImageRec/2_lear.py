# 参考：https://toukei-lab.com/python-image
import matplotlib.pyplot as plt
import glob
import cv2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
import os

#データ準備
run_dir = os.path.dirname(__file__)
files = glob.glob(run_dir + "images/*.jpg")
files = sorted(files)
df_label = pd.read_csv(run_dir + "/train.csv")

file_list = []
for file in files:
  file = cv2.imread(file)
  file_list.append(file)

#画素値を正規化
file_list = [file.astype(float)/255 for file in file_list] 
train_x, valid_x, train_y, valid_y = train_test_split(file_list, df_label, test_size=0.2)

# train_y, valid_y をダミー変数化
train_y = to_categorical(train_y["gender_status"])
valid_y = to_categorical(valid_y["gender_status"])

# リスト型を配列型に
train_x = np.array(train_x)
valid_x = np.array(valid_x)

#層の定義
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(8, activation='softmax'))

# モデルを構築
model.compile(optimizer=tf.optimizers.Adam(0.01), loss='categorical_crossentropy', metrics=['accuracy'])

# Early stoppingを適用してフィッティング
log = model.fit(train_x, train_y, epochs=100, batch_size=10, verbose=True,
                callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                     min_delta=0, patience=10, 
                                                         verbose=1)],
                validation_data=(valid_x, valid_y))
