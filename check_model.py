import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os
import pandas as pd

class SelfBoostedAttention(tf.keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        super(SelfBoostedAttention, self).__init__(**kwargs)
        self.attention = tf.keras.layers.DepthwiseConv2D((3, 3), activation='sigmoid', padding='same')

    def call(self, inputs):
        attn_map = self.attention(inputs)
        return inputs * attn_map + inputs

    def get_config(self):
        config = super().get_config()
        config.update({"filters": 128})
        return config

model = tf.keras.models.load_model('alzheimer_detection_model.h5', custom_objects={'SelfBoostedAttention': SelfBoostedAttention}, compile=False)

class_indices = {'Mild Dementia': 0, 'Moderate Dementia': 1, 'Non Demented': 2, 'Very mild Dementia': 3}
class_names = {v: k for k, v in class_indices.items()}

image_dir = input("Введіть шлях до директорії із зображеннями: ")

if not os.path.exists(image_dir):
    print("Вказана директорія не існує!")
    exit()

correct_predictions = {class_name: 0 for class_name in class_indices.keys()}
total_images = {class_name: 0 for class_name in class_indices.keys()}

for subdir, _, files in os.walk(image_dir):
    true_class_name = os.path.basename(subdir)
    if true_class_name not in class_indices:
        continue

    for file in files:
        img_path = os.path.join(subdir, file)

        img = image.load_img(img_path, target_size=(96, 96))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions[0])
        predicted_class_name = class_names[predicted_class_index]

        total_images[true_class_name] += 1
        if predicted_class_name == true_class_name:
            correct_predictions[true_class_name] += 1

accuracy_results = []
for class_name in class_indices.keys():
    total = total_images[class_name]
    correct = correct_predictions[class_name]
    accuracy = (correct / total * 100) if total > 0 else 0
    accuracy_results.append([class_name, correct, total, round(accuracy, 2)])

df = pd.DataFrame(accuracy_results, columns=["Class", "Correct predictions", "Total amount", "Accuracy (%)"])
print(df)

df.to_csv("classification_results.csv", index=False)
print("Результати збережені у classification_results.csv")