"""
This model catagorizes single peaked microbursts that were first 
identified using the O'Brien et al.'s, 2003 burst parameter.
"""
import pathlib

import tensorflow as tf
import pandas as pd
import numpy as np

import microburst_ann.config as config

# Load the data
train_path = pathlib.Path(config.PROJECT_DIR, 'data', 'train.csv')
test_path = pathlib.Path(config.PROJECT_DIR, 'data', 'test.csv')

train_df = pd.read_csv(train_path, index_col=0)
test_df = pd.read_csv(test_path, index_col=0)

train_df.dropna(inplace=True)
test_df.dropna(inplace=True)

train_labels = train_df.pop('label')
test_labels = test_df.pop('label')

train_dataset = tf.data.Dataset.from_tensor_slices(
    (train_df.to_numpy(), train_labels.to_numpy())
    )
test_dataset = tf.data.Dataset.from_tensor_slices(
    (test_df.to_numpy(), test_labels.to_numpy())
    )
shuffled_train_dataset = train_dataset.shuffle(train_df.shape[0]).batch(1)


# # Model
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(50,)),
  tf.keras.layers.Dense(10, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(1),
  tf.keras.layers.Softmax(),
])

model.compile(optimizer='adam',
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=['accuracy'])
print(model.summary())

print(model(train_df.iloc[0].to_numpy().reshape((1, 50))))

history = model.fit(shuffled_train_dataset, epochs=3)

print(model.evaluate(test_df.to_numpy(), test_labels, verbose=2))