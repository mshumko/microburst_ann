"""
This model catagorizes single peaked microbursts that were first 
identified using the O'Brien et al.'s, 2003 burst parameter.
"""
import pathlib

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
  tf.keras.layers.Dense(25, activation='relu'),
  tf.keras.layers.Dense(10, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(1, activation='sigmoid'),
])

model.compile(optimizer='adam',
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=['accuracy'])
print(model.summary())

# print(model(train_df.iloc[0].to_numpy().reshape((1, 50))))

history = model.fit(shuffled_train_dataset, 
                    validation_data=(test_df.to_numpy(), test_labels), 
                    epochs=3)

# i = 10; print(model(train_df.iloc[i].to_numpy().reshape((1, 50))), train_labels.iloc[i])

# print(model.evaluate(test_df.to_numpy(), test_labels, verbose=2))

# n_correct=0
# for i in range(test_df.shape[0]):
#     if round(model(test_df.iloc[i].to_numpy().reshape((1, 50))).numpy()[0][0]) == test_labels.iloc[i]:
#         n_correct += 1
# print(f'n_correct={n_correct}, n_correct/n_total={n_correct/test_df.shape[0]}')

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1) # set the vertical range to [0-1]
plt.show()