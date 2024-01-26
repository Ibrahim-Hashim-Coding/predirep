'''
Generate layer specific unit representations for PrediRep on dataset.
'''
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import numpy as np
from keras.models import Model, model_from_json
from keras.layers import Input
from data_utils import SequenceGenerator
from data_settings import *
import glob
from keras import backend as K
from networks.predirep import PrediRep

nt = 10  # for how many time steps to extract unit information
layers = [0, 1, 2, 3]  # for which layers to extract unit information

# Load test data and create generator
test_file = os.path.join(DATA_DIR, 'X_example.hkl')
test_sources = os.path.join(DATA_DIR, 'sources_example.hkl')
test_generator = SequenceGenerator(test_file, test_sources, nt, sequence_start_mode='unique',
                                   data_format="channels_last", N_seq=1)
X_test = test_generator.create_all()

# Create folder to store images
if not os.path.exists(WEIGHTS_DIR + 'unit_rep'):
    os.mkdir(WEIGHTS_DIR + 'unit_rep')

# Load weights and model
weights_file = WEIGHTS_DIR + 'predirep_weights.hdf5'
json_file = WEIGHTS_DIR + 'predirep_model.json'
f = open(json_file, 'r')
json_string = f.read()
f.close()

train_model = model_from_json(json_string, custom_objects={'PrediRep': PrediRep})
train_model.load_weights(weights_file)

input_shape = list(train_model.layers[0].batch_input_shape[1:])
input_shape[0] = nt
inputs = Input(shape=tuple(input_shape))

layer_config = train_model.layers[1].get_config()

# Generate unit representations for all layers
for layer_c, layer in enumerate(layers):
    for unit_c, unit in enumerate(['{}{}'.format(x, layer) for x in ['R', 'Ahat', 'A', 'E']]):
        layer_config['output_mode'] = unit
        test_predirep = PrediRep(weights=train_model.layers[1].get_weights(), **layer_config)
        predictions = test_predirep(inputs)
        test_model = Model(inputs=inputs, outputs=predictions)
        X_hat = test_model.predict(X_test)
        np.save(WEIGHTS_DIR + '/unit_rep/{}.npy'.format(unit), X_hat)
