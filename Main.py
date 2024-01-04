from prep_data import set_train_data
from build_model import build_model, train_model
from visualize import visualize_history
import os
import numpy as np


data_dir = '../data'
train_data_dir = os.path.join(data_dir, 'Train')

num_classes = 43

x_train, y_train, x_val, y_val = set_train_data(
    train_data_dir, num_classes)
print(x_train.shape)
print(y_train.shape)
print(x_val.shape)
print(y_val.shape)
input_shape = x_train.shape[1:]

model = build_model(input_shape, num_classes)
model, history = train_model(model, x_train, y_train, x_val, y_val)
model = build_model((64,64,3),43)
model.summary()
#visualize_history(history)

# Eğitim ve validasyon işlenmiş verilerinin kaydedilmesi
processed_data_dir = 'processed_data'

np.save(processed_data_dir+'/x_train.npy', x_train)
np.save(processed_data_dir+'/y_train.npy', y_train)
np.save(processed_data_dir+'/x_val.npy', x_val)
np.save(processed_data_dir+'/y_val.npy', y_val)

# Modelin kaydedilmesi
model.save('models/model.h5', save_format='h5')