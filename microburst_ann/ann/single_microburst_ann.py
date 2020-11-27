import tensorflow as tf
import pandas as pd


# Load and normalize the data


# Model
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(50)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(1),
])
tf.keras.losses

model.compile(optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])
print(model.summary())

model.fit(x_train, y_train, epochs=5)

print(model.evaluate(x_test,  y_test, verbose=2))