#!/usr/bin/env python3

import numpy as np

#set up so the data can be replicated
np.random.seed(0)

#function generates points on a sin wave of given frequency and phase
def sin_wave(time, frequency, phase):

    return np.sin(2 * np.pi * frequency * time + phase)

#function generates random sin waves with added noise
def generate_true_data():
    time = np.linspace(0, 100, num = 1024)
    #frequencies based on the range of CWs
    frequency = np.random.uniform(low=20, high=2000)
    phase = np.random.uniform(low=0, high=np.pi*2)

    data = sin_wave(time, frequency, phase)
    #add gausian noise
    data_noisy = data + np.random.normal(0, 0.5, data.shape)
    return data_noisy

#function generates random gaussian noise
def generate_false_data():
    data = np.zeros(1024, dtype=float)
    data_noisy = data + np.random.normal(0, 0.5, data.shape)
    return data_noisy

training_data_array = np.array([generate_true_data()])
training_label_array = np.array([1])

#generate training data
for x in range(8000):
    data_type = np.random.randint(0,2)
    if data_type == 1:
        training_data_array = np.vstack((np.array([generate_true_data()]), training_data_array))
        training_label_array = np.vstack((np.array([1]), training_label_array))
    if data_type == 0:
        training_data_array = np.vstack((np.array([generate_false_data()]), training_data_array))
        training_label_array = np.vstack((np.array([0]), training_label_array))
    x+=1

#save training data
np.save("training-data.npy", training_data_array)
np.save("training-labels.npy", training_label_array)

testing_data_array = np.array([generate_true_data()])
testing_label_array = np.array([1])

#generate testing data
for x in range(2000):
    data_type = np.random.randint(0,2)
    if data_type == 1:
        testing_data_array = np.vstack((np.array([generate_true_data()]), testing_data_array))
        testing_label_array = np.vstack((np.array([1]), testing_label_array))
    if data_type == 0:
        testing_data_array = np.vstack((np.array([generate_false_data()]), testing_data_array))
        testing_label_array = np.vstack((np.array([0]), testing_label_array))
    x+=1

#save testing data
np.save("testing-data.npy", testing_data_array)
np.save("testing-labels.npy", testing_label_array)
