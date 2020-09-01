import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas

df = pandas.read_parquet('user-accounts.parquet')
df = df[df['balance_account_2'] >= 0]
df = df[df['balance_account_1'] >= 0]
df = df[df['balance_account_0'] >= 0]

a = df['balance_account_0'].to_numpy()
b = df['balance_account_1'].to_numpy()
c = df['balance_account_2'].to_numpy()

train_data = np.array([a, b]).T

model = keras.Sequential([
    layers.Dense(4, activation='relu', input_shape=[2]),
    layers.Dense(2, activation='relu'),
    layers.Dense(1)
])

optimizer = tf.keras.optimizers.RMSprop(0.001)

model.compile(loss='mse',
              optimizer=optimizer,
              metrics=['mae', 'mse'])

model.fit(train_data, c, epochs=500)

print(model.predict([[1000, 5]]))
