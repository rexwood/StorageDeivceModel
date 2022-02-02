import numpy as np
import pandas as pd
import tensorflow as tf
import os
from model import storageNet

input_dir = './dataset_7/input'
output_dir = './dataset_7/output'

input_data = []
output_data = []

for data_file in os.listdir(input_dir):
        # print(data_file)
        input = pd.read_csv(os.path.join(input_dir, data_file), header=None).values
        output = pd.read_csv(os.path.join(output_dir, data_file), header=None).values
        input_data.extend(input)
        output_data.extend(output)

input_data = np.asarray(input_data)
output_data = np.asarray(output_data)

print(input_data.shape)
print(output_data.shape)


np.random.seed(1)
np.random.shuffle(input_data)
np.random.seed(1)
np.random.shuffle(output_data)

num_train_entries = int(input_data.shape[0] * 0.80)
print('num train entries: ', num_train_entries)
train_X = input_data[:num_train_entries]
train_y = output_data[:num_train_entries]
test_X = input_data[num_train_entries:]
test_y = output_data[num_train_entries:]

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation='relu', input_dim=108),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(7)
])
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.mean_squared_error,
    metrics=['mae', 'mse'],
)
# model = storageNet.StorageNet()
# model.compile(
#     optimizer=tf.keras.optimizers.Adam(0.001),
#     loss=tf.keras.losses.mean_squared_error,
#     metrics=['mae', 'mse'],
# )


mse_losses = []

for i in range(1000):
    model.fit(train_X, train_y, epochs=1, batch_size=128)
    res = model.evaluate(test_X, test_y)
    mse_losses.append(res[0])

print("min test loss:", np.min(mse_losses))
import matplotlib.pyplot as plt
plt.xlabel('epochs')
plt.ylabel('MSE Loss')
plt.title('test result')
plt.plot(mse_losses)
plt.show()

# model.save_weights('./weights/saved_model_7.h5')
model.save("weights")