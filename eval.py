# Multi step evaluation:
import numpy as np
import pandas as pd
import tensorflow as tf

def test_profile(model, raw_io_info, first_prev):
    for i in range(raw_io_info.shape[0]):
        if i == 0:
            temp_latency = model.predict(first_prev)
        else:
            print(temp_input.shape)
            temp_latency = model.predict(temp_input)
        temp_input = np.hstack((raw_io_info[i:i+1], temp_latency))
    return temp_input

# Load the model:
model = tf.keras.models.load_model("weights")

# test the whole profile:
test_input = pd.read_csv('./dataset_7/input/data_azure_.csv', header=None).values
first_prev = test_input[0:1]
test_input = test_input[:, :101]
# print(test_input.shape)
# print(first_prev.shape)
print(test_profile(model, test_input, first_prev)[:, -7:])