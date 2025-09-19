import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import sys

model = tf.keras.models.load_model('matar_paneer_model.h5')

class_names = ['matar_paneer', 'not_matar_paneer']

img_path = 'test.jpeg' 

img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)[0][0]
predicted_class = class_names[int(prediction > 0.5)]
print("Prediction: This image is", predicted_class.upper())
