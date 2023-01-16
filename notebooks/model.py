import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import mixed_precision

mixed_precision.set_global_policy(policy="mixed_float16")

EPOHCS = 50
train_generator = tf.keras.utils.image_dataset_from_directory(
    '../data/train',
    image_size=(224, 224),
    validation_split=0.2,
    subset='training',
    seed=123,
    batch_size=32
)
test_generator = tf.keras.utils.image_dataset_from_directory(
    '../data/test',
    image_size=(224, 224),
    validation_split=0.2,
    subset='validation',
    seed=123,
    batch_size=32
)
# model = tf.keras.Sequential([
#   tf.keras.layers.Conv2D(64, (4, 4), padding='Same',
#                          activation='relu', input_shape=(150, 150, 3)),
#   tf.keras.layers.MaxPool2D(2, 2),

#   tf.keras.layers.Conv2D(128, (4, 4), padding='Same', activation='relu'),
#   tf.keras.layers.MaxPool2D(2, 2),

#   tf.keras.layers.Conv2D(128, (4, 4), padding='Same', activation='relu'),
#   tf.keras.layers.MaxPool2D(2, 2),

#   tf.keras.layers.Conv2D(128, (4, 4), padding='Same', activation='relu'),
#   tf.keras.layers.MaxPool2D(2, 2),

#   tf.keras.layers.Conv2D(64, (4, 4), padding='Same',
#                          activation='relu', input_shape=(150, 150, 3)),

#   tf.keras.layers.MaxPool2D(2, 2),
#   tf.keras.layers.Flatten(),
#   tf.keras.layers.Dense(512, activation='relu'),
#   tf.keras.layers.Dense(120)
# ])
base_model = tf.keras.applications.EfficientNetB0(include_top=False)
base_model.trainable = False

# Create Functional model
inputs = layers.Input(shape=(224, 224, 3), name="input_layer")
# Note: EfficientNetBX models have rescaling built-in but if your model didn't you could have a layer like below
# x = layers.Rescaling(1./255)(x)
x = base_model(inputs, training=False)  # set base_model to inference mode only
x = layers.GlobalAveragePooling2D(name="pooling_layer")(x)
x = layers.Dense(120)(x)  # want one output neuron per class
# Separate activation of output layer so we can output float32 activations
outputs = layers.Activation(
    "softmax", dtype=tf.float32, name="softmax_float32")(x)
model = tf.keras.Model(inputs, outputs)

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=EPOHCS
)
