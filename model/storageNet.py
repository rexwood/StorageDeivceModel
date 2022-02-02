import tensorflow as tf

from tensorflow.keras import layers

class StorageNet(tf.keras.Model):

    def __init__(self):
        super(StorageNet, self).__init__()
        self.dense_1 = layers.Dense(256, activation=tf.nn.relu, input_dim=108)
        self.dense_2 = layers.Dense(512, activation=tf.nn.relu)
        self.dense_3 = layers.Dense(512, activation=tf.nn.relu)
        self.dense_4 = layers.Dense(512, activation=tf.nn.relu)
        self.dense_5 = layers.Dense(256, activation=tf.nn.relu)
        self.dense_6 = layers.Dense(7)

    def call(self, inputs):
        # Define your forward pass here,
        # using layers you previously defined (in `__init__`).
        x = self.dense_1(inputs)
        x = self.dense_2(x)
        x = self.dense_3(x)
        x = self.dense_4(x)
        x = self.dense_5(x)
        return self.dense_6(x)

# def storageNet():
#     model = tf.keras.models.Sequential([
#         tf.keras.layers.Dense(256, activation='relu', input_dim=108),
#         tf.keras.layers.Dense(512, activation='relu'),
#         tf.keras.layers.Dense(512, activation='relu'),
#         tf.keras.layers.Dense(512, activation='relu'),
#         tf.keras.layers.Dense(256, activation='relu'),
#         tf.keras.layers.Dense(7)
#     ])
# model.compile(
#     optimizer=tf.keras.optimizers.Adam(0.001),
#     loss=tf.keras.losses.mean_squared_error,
#     metrics=['mae', 'mse'],
# )
