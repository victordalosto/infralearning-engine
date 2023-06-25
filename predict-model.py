import os
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from domain.Dados import Dados
from tensorflow.keras.preprocessing.image import load_img, img_to_array

path = 'D:\\imgs\\mount\\FULL\\advertencia\\'
labels = ['advertencia', 'educativa', 'indicativa', 'regulamentacao', 'servicos', 'temporaria', 'turistico']

model = tf.keras.models.load_model('models\\classification.h5')

for img in os.listdir(path):
    image_path = os.path.join(path, img)
    image = load_img(image_path, target_size=(256, 256))
    image_array = img_to_array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    predictions = model.predict(image_array)
    predicted_label = labels[np.argmax(predictions)]
    confidence = np.round(predictions[0, np.argmax(predictions)] * 100, decimals=1)
    print('Predict:', predicted_label, '(', confidence, '%)')
