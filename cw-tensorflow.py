#!/usr/bin/env python3

#TensorFlow and tf.keras
import tensorflow as tf

#helper libraries
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

#set up path to directory, replace with path on your computer
p = Path('/home/shira/Documents/icr/cw-dnn/')

#load data
training_data = np.load(p / "training-data.npy")
training_labels = np.load(p / "training-labels.npy")
testing_data = np.load(p / "testing-data.npy")
testing_labels = np.load(p / "testing-labels.npy")

class_names = ['no_wave', 'wave']

#set up the layers
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(2)
])

#compile the model
model.compile(optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

#train the model
model.fit(training_data, training_labels, epochs=10)

#test the accuracy
test_loss, test_acc = model.evaluate(testing_data, testing_labels, verbose=2)
print('\nTest accuracy:', test_acc)

#examine the predictions
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(testing_data)

print(predictions[0])
print(np.argmax(predictions[0]))
print(testing_labels[0])

