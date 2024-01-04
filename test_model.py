from prep_data import set_test_data
from tensorflow.keras.models import load_model
import os
import numpy as np


data_dir = '../data'
test_data_dir = 'C:\\Users\\rumeysa\\OneDrive\\Masaüstü\\traffic_sign\\data\\Test.csv'

num_classes = 43

# Test verilerinin okunması
x_test, y_test = set_test_data(data_dir, test_data_dir, num_classes)

model_dir = 'C:\\Users\\rumeysa\\OneDrive\\Masaüstü\\traffic_sign\\traffic_sign\\models\\model.h5'
model = load_model(model_dir)

model.evaluate(x_test, y_test)

processed_data_dir = 'processed_data'
np.save(processed_data_dir+'/x_test.npy', x_test) 
np.save(processed_data_dir+'/y_test.npy', y_test)