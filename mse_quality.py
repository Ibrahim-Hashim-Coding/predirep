'''
Calculate MSE for PrediRep.
'''
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from keras.models import Model, model_from_json
from keras.layers import Input
from data_utils import SequenceGenerator
from data_settings import *
import numpy as np
from keras import backend as K
import os
from networks.predirep import PrediRep  # change this if you want to use other networks

# Set parameters
nt = 10  # number of time steps to calculate MSE over
learn_type = "equal"  # the learn type of the model, either zero, all or equal

# Load test files and create generator and dataset
test_file = os.path.join(DATA_DIR, 'X_example.hkl')
test_sources = os.path.join(DATA_DIR, 'sources_example.hkl')
test_generator = SequenceGenerator(test_file, test_sources, nt, sequence_start_mode='unique',
                                   data_format="channels_last")
X_test = test_generator.create_all()

# Calculate MSE if copying last frame
last_frame_mean = np.mean((X_test[:, :-1] - X_test[:, 1:])**2)

# Load weights and model
weights_file = WEIGHTS_DIR + 'predirep_{}_weights.hdf5'.format(learn_type)
json_file = WEIGHTS_DIR + 'predirep_{}_model.json'.format(learn_type)

f = open(json_file, 'r')
json_string = f.read()
f.close()

train_model = model_from_json(json_string, custom_objects={'PrediRep': PrediRep})
train_model.load_weights(weights_file)

input_shape = list(train_model.layers[0].batch_input_shape[1:])
input_shape[0] = nt
inputs = Input(shape=tuple(input_shape))

layer_config = train_model.layers[1].get_config()

# Create testing model (to output predictions)
layer_config['output_mode'] = 'Ahat0'
test_predirep = PrediRep(weights=train_model.layers[1].get_weights(), **layer_config)
predictions = test_predirep(inputs)
test_model = Model(inputs=inputs, outputs=predictions)
X_hat = test_model.predict(X_test)

# Calculate and output model mse
mse_mean = np.mean((X_hat[:, 1:] - X_test[:, 1:]) ** 2)

print("Model MSE is: {}".format(mse_mean))
print("Last Frame MSE is: {}".format(last_frame_mean))
