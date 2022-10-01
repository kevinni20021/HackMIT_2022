import numpy as np
import os
import tensorflow as tf

batch_size=32
img_height=64
img_width=64

base_dir = os.path.join(os.getcwd(), "./ASL_dataset/")
train_dir = os.path.join(base_dir, 'asl_alphabet_train/asl_alphabet_train')
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    validation_split=0.1,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)
