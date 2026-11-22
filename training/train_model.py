import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow.keras.layers as tfl

from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip, RandomRotation

BATCH_SIZE = 32
IMG_SIZE = (256, 256)
directory = "images"
train_dataset = image_dataset_from_directory(directory + "/train/",
                                             shuffle=True,
                                             batch_size=BATCH_SIZE,
                                             image_size=IMG_SIZE)

validation_dataset = image_dataset_from_directory(directory + "/val/",
                                             shuffle=True,
                                             batch_size=BATCH_SIZE,
                                             image_size=IMG_SIZE)


class_names = train_dataset.class_names

AUTOTUNE = tf.data.experimental.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)

def data_augmenter():
    data_augmentation = tf.keras.Sequential()
    data_augmentation.add(RandomFlip('horizontal'))
    data_augmentation.add(RandomRotation(0.2))
    
    return data_augmentation

data_augmentation = data_augmenter()

preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
IMG_SHAPE = IMG_SIZE + (3,)

def recycle_model(image_shape=IMG_SIZE, data_augmentation=data_augmenter()):
    """Transfer-learning classifier: frozen MobileNetV2 base with a new head (one logit per class)."""
    input_shape = image_shape + (3,)
    
    base_model_path="imagenet_base_model/without_top_mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_160_no_top.h5"
    
    base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape,
                                                   include_top=False,
                                                   weights=base_model_path)
    
    base_model.trainable = False 

    inputs = tf.keras.Input(shape=input_shape) 
    
    x = data_augmentation(inputs)
    
    x = preprocess_input(x) 
    
    # keep batch-norm layers in inference mode
    x = base_model(x, training=False) 
    
    x = tfl.GlobalAveragePooling2D()(x) 
    x = tfl.Dropout(0.2)(x)
        
    outputs = tfl.Dense(units=len(class_names))(x)
    
    model = tf.keras.Model(inputs, outputs)
    
    return model

mdl = recycle_model(IMG_SIZE, data_augmentation)
mdl.summary()
base_learning_rate = 0.001
mdl.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
          metrics=['accuracy'])

initial_epochs = 5
history = mdl.fit(train_dataset, validation_data=validation_dataset, epochs=initial_epochs, verbose=2)
acc = [0.] + history.history['accuracy']
val_acc = [0.] + history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()
