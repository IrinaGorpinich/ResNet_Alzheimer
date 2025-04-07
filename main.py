import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.losses import Loss
from tensorflow.keras.applications import ResNet50
import numpy as np
import os
import shutil
import random
from datetime import datetime

log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
os.makedirs(log_dir, exist_ok=True)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

data_dir = "/Users/gorpinicirina/Documents/Data"
output_dir = "/Users/gorpinicirina/Documents/Split_Data"

train_ratio = 0.6
val_ratio = 0.2
test_ratio = 0.2

def is_already_filled(split_dir):
    if not os.path.exists(split_dir):
        return False
    for category in os.listdir(split_dir):
        category_path = os.path.join(split_dir, category)
        if os.path.isdir(category_path) and os.listdir(category_path):
            return True
    return False

if all(is_already_filled(os.path.join(output_dir, split)) for split in ["train", "val", "test"]):
    print("Дані вже розділені, пропускаємо копіювання!")
else:
    for split in ["train", "val", "test"]:
        for category in os.listdir(data_dir):
            os.makedirs(os.path.join(output_dir, split, category), exist_ok=True)
    for category in os.listdir(data_dir):
        category_path = os.path.join(data_dir, category)
        if not os.path.isdir(category_path):
            continue
        images = os.listdir(category_path)
        random.shuffle(images)

        train_split = int(len(images) * train_ratio)
        val_split = int(len(images) * (train_ratio + val_ratio))

        for i, img in enumerate(images):
            src_path = os.path.join(category_path, img)

            if i < train_split:
                dst_path = os.path.join(output_dir, "train", category, img)
            elif i < val_split:
                dst_path = os.path.join(output_dir, "val", category, img)
            else:
                dst_path = os.path.join(output_dir, "test", category, img)

            shutil.copy(src_path, dst_path)

        print("Дані успішно розділені на train/val/test!")

class TensorBoardBatchCallback(tf.keras.callbacks.Callback):
    def __init__(self, log_dir, update_frequency=0.25):
        super().__init__()
        self.log_dir = log_dir
        self.update_frequency = update_frequency
        self.writer = tf.summary.create_file_writer(log_dir)
        self.step = 0

    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}
        total_batches = self.params['steps']
        update_interval = int(total_batches * self.update_frequency)

        if batch % update_interval == 0 or batch == total_batches - 1:
            with self.writer.as_default():
                for metric, value in logs.items():
                    tf.summary.scalar(f'train/{metric}', value, step=self.step)
            self.step += 1

    def on_test_batch_end(self, batch, logs=None):
        logs = logs or {}
        total_batches = self.params.get('validation_steps', 0)
        if total_batches == 0:
            return
        update_interval = int(total_batches * self.update_frequency)

        if batch % update_interval == 0 or batch == total_batches - 1:
            with self.writer.as_default():
                for metric, value in logs.items():
                    tf.summary.scalar(f'val/{metric}', value, step=self.step)
            self.step += 1

class SelfBoostedAttention(layers.Layer):
      def __init__(self, filters):
          super(SelfBoostedAttention, self).__init__()
          self.attention = layers.DepthwiseConv2D((3, 3), activation='sigmoid', padding='same')

      def call(self, inputs):
          attn_map = self.attention(inputs)
          return inputs * attn_map + inputs


def build_model(input_shape=(96, 96, 3), num_classes=4):
    base_model = ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)
    for layer in base_model.layers[:-5]:
        layer.trainable = False
    x = base_model.output
    x = SelfBoostedAttention(128)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return models.Model(inputs=base_model.input, outputs=outputs)

class DynamicClassWeights(tf.keras.callbacks.Callback):
    def __init__(self, train_generator):
        super().__init__()
        self.train_generator = train_generator

    def on_epoch_end(self, epoch, logs=None):
        min_weight = 0.1
        max_weight = 10
        class_counts = np.bincount(self.train_generator.classes)
        total = np.sum(class_counts)
        new_weights = {i: np.clip(total / (len(class_counts) * count), min_weight, max_weight) for i, count in enumerate(class_counts) if count > 0}
        self.model.loss.class_weights = tf.convert_to_tensor([new_weights[i] for i in sorted(new_weights.keys())], dtype=tf.float32)


class CustomLoss(Loss):
    def __init__(self, class_weights):
        super(CustomLoss, self).__init__()
        weights_list = [class_weights[i] for i in sorted(class_weights.keys())]
        self.class_weights = tf.convert_to_tensor(weights_list, dtype=tf.float32)
    def call(self, y_true, y_pred):
        weights = tf.reduce_sum(self.class_weights * y_true, axis=1)
        loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        return tf.reduce_mean(weights * loss)


datagen = ImageDataGenerator(
    rescale=1./255
)

train_generator = datagen.flow_from_directory(
    os.path.join(output_dir, "train"),
    target_size=(96, 96),
    batch_size=32,
    class_mode='categorical',
    subset='training',
    shuffle=True
)


val_generator = datagen.flow_from_directory(
    os.path.join(output_dir, "val"),
    target_size=(96, 96),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

test_generator = datagen.flow_from_directory(
    os.path.join(output_dir, "test"),
    target_size=(96, 96),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

class HardSampleMining(tf.keras.callbacks.Callback):
    def __init__(self, train_generator, factor=1.2):
        super().__init__()
        self.train_generator = train_generator
        self.factor = factor

    def on_epoch_end(self, epoch, logs=None):
        predictions = self.model.predict(self.train_generator)
        true_labels = self.train_generator.classes
        predicted_labels = np.argmax(predictions, axis=1)

        errors = true_labels != predicted_labels
        hard_samples = np.array(self.train_generator.filenames)[errors]

        print(f"Кількість важливих зразків після {epoch + 1} епохи: {len(hard_samples)}")

class_weights = {i: 1.0 for i in range(4)}

model = build_model()

optimizer = optimizers.Adam(learning_rate=0.00001)

model.compile(
    optimizer=optimizer,
    loss=CustomLoss(class_weights),
    metrics=['accuracy']
)

tensorboard_batch_callback = TensorBoardBatchCallback(log_dir=log_dir)
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=50,
    callbacks=[DynamicClassWeights(train_generator), HardSampleMining(train_generator), tensorboard_callback, tensorboard_batch_callback],
    steps_per_epoch=max(1, train_generator.samples // train_generator.batch_size),
    validation_steps=max(1, val_generator.samples // val_generator.batch_size)
)

model.save('alzheimer_detection_model.h5')

test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test accuracy: {test_accuracy}")