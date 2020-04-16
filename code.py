# -*- coding: utf-8 -*-
"""
@author: Christofer Stoll
"""

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import glob 
import cv2
#import os
 
from sklearn.model_selection import train_test_split
from skimage.color import rgb2grey

NUM_CLASSES = 43
np.random.seed(42)

# Pfad zu den Trainingsdaten
data_path = "src/GTSRB/Final_Training/Images"
 
images = []
image_labels = []
 
# Pfade zu den einzelnen Bildern
for i in range(NUM_CLASSES):
    image_path = data_path + "/" + format(i, "05d") + "/"
    for img in glob.glob(image_path + "*.ppm"):
        image = cv2.imread(img)
        image = rgb2grey(image) # Umwandlung in Graustufen
        image = (image / 255.0) # Neu skalieren
        image = cv2.resize(image, (32, 32)) #Größe vereinheitlichen
        images.append(image)
        
        # Erstellung der Label für die Bilder und Transfer in eine Binär-Matrix (1-aus-n-Code)
        labels = np.zeros((NUM_CLASSES, ), dtype=np.float32)
        labels[i] = 1.0
        image_labels.append(labels)
 
images = np.stack([img[:, :, np.newaxis] for img in images], axis=0).astype(np.float32)
image_labels = np.matrix(image_labels).astype(np.float32)


plt.imshow(images[45, :, :, :].reshape(32, 32), cmap="gray")
print(image_labels[45, :])

print(images.shape)
print(len(images))

# Aufteilung in Treinings- und Testsets
(train_X, test_X, train_y, test_y) = train_test_split(images, image_labels, 
                                                      test_size=0.2, 
                                                      random_state=42)
print(train_X.shape)
print(train_y.shape)
print(test_X.shape)
print(test_y.shape)

# Drei Conv2D() (Dreidimensionale Faltung) Layer mit den Dimensionen 32, 64 und 128
model = tf.keras.models.Sequential()
input_shape = (32, 32, 1) # Bilder mit der Auflösung von 32x32 Pixel und Graustufe

model.add(tf.keras.layers.Conv2D(32, (5, 5), padding='same', activation='relu', input_shape=input_shape, data_format="channels_last"))
model.add(tf.keras.layers.BatchNormalization(axis=-1))      
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(0.2))
        
model.add(tf.keras.layers.Conv2D(64, (5, 5), padding='same', activation='relu', data_format="channels_last"))
model.add(tf.keras.layers.BatchNormalization(axis=-1))

model.add(tf.keras.layers.Conv2D(128, (5, 5), padding='same', activation='relu', data_format="channels_last"))
model.add(tf.keras.layers.BatchNormalization(axis=-1))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(0.2))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(512, activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.4))

model.add(tf.keras.layers.Dense(43, activation='softmax'))

optimizer = tf.keras.optimizers.Adam(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, 
              metrics=['accuracy'])

history = model.fit(train_X, train_y, validation_data=(test_X, test_y),epochs=10)

#Sichern des Modells
model.save('Gespeichertes Modell 1.h5') #Modelnamen bei jedem Durchgang abändern (Sonst überschreibt man die Datei)

#Plot für Diagramm
num_epochs = np.arange(0, 10)
plt.figure(dpi=300)
plt.plot(num_epochs, history.history['loss'], label='train_loss', c='blue')
plt.plot(num_epochs, history.history['val_loss'], label='val_loss', c='red')
plt.plot(num_epochs, history.history['acc'], label='train_acc', c='green')
plt.plot(num_epochs, history.history['val_acc'], label='val_acc', c='yellow')
plt.title('Wert der Verlustfunktion')
plt.xlabel('Epochen')
plt.ylabel('Genauigkeit bzw. Verlust')
plt.legend()
plt.savefig('Diagramm.png')



