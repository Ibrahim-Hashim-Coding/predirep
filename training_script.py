'''
Train networks on video sequences.
Code is built around that of PredNet (Lotter et al., 2016. - https://coxlab.github.io/prednet/)
'''

import os
import numpy as np
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense, Flatten, TimeDistributed
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from data_utils import SequenceGenerator
from data_settings import *
from networks.predirep import PrediRep

# Data files
train_file = os.path.join(DATA_DIR, 'X_train.hkl')
train_sources = os.path.join(DATA_DIR, 'sources_train.hkl')
val_file = os.path.join(DATA_DIR, 'X_val.hkl')
val_sources = os.path.join(DATA_DIR, 'sources_val.hkl')

# Training parameters
nb_epoch = 150
batch_size = 4
samples_per_epoch = 500
N_seq_val = 100  # number of sequences to use for validation
learn_type = "all"  # how error layers contribute to loss
nt = 10  # number of time steps

# Model parameters
n_channels, im_height, im_width = (3, 128, 160)
input_shape = (n_channels, im_height, im_width) if K.image_data_format() == 'channels_first' else (
im_height, im_width, n_channels)
stack_sizes = (n_channels, 8, 16, 32)
R_stack_sizes = (8, 16, 32, 64)
Ahat_filt_sizes = (3, 3, 3, 3)
R_filt_sizes = (3, 3, 3, 3)

# Loss parameters
layer_loss_dict = {
    "zero": np.array([1., 0., 0., 0.]),
    "all": np.array([1., 0.1, 0.1, 0.1]),
    "equal": np.array([1., 1., 1., 1.]),
}

layer_loss_weights = layer_loss_dict[learn_type]
layer_loss_weights = np.expand_dims(layer_loss_weights, 1)
time_loss_weights = 1. / (nt - 1) * np.ones((nt, 1))  # equally weight all timesteps except the first
time_loss_weights[0] = 0

# Set up model saving
save_model = True  # if model and weights will be saved
weights_file = os.path.join(WEIGHTS_DIR, 'predirep_weights.hdf5')  # where weights will be saved
json_file = os.path.join(WEIGHTS_DIR, 'predirep_model.json')  # where model will be saved

# Set up the network - some parameters set as default, can be changed see the predirep.py code under networks
predirep = PrediRep(stack_sizes, R_stack_sizes, Ahat_filt_sizes, R_filt_sizes, return_sequences=True)

# Define the loss and generate the model
inputs = Input(shape=(nt,) + input_shape)
errors = predirep(inputs)  # errors will be (batch_size, nt, nb_layers)
errors_by_time = TimeDistributed(Dense(1, trainable=False), weights=[layer_loss_weights, np.zeros(1)], trainable=False)(
    errors)  # calculate weighted error by layer
errors_by_time = Flatten()(errors_by_time)  # will be (batch_size, nt)
final_errors = Dense(1, weights=[time_loss_weights, np.zeros(1)], trainable=False)(
    errors_by_time)  # weight errors by time

model = Model(inputs=inputs, outputs=final_errors)
model.compile(loss='mean_absolute_error', optimizer='adam')
print(model.summary())

# Create data generators
train_generator = SequenceGenerator(train_file, train_sources, nt, batch_size=batch_size, shuffle=True)
val_generator = SequenceGenerator(val_file, val_sources, nt, batch_size=batch_size, N_seq=N_seq_val)

# Create learning rate schedule
lr_schedule = lambda epoch: 0.001 if epoch < 75 else 0.0001  # start with lr of 0.001 and reduce by 10 after 75 epochs
callbacks = [LearningRateScheduler(lr_schedule)]

# Create location to save model and weights, set up callbacks for saving the weights
if save_model:
    if not os.path.exists(WEIGHTS_DIR): os.mkdir(WEIGHTS_DIR)
    callbacks.append(ModelCheckpoint(filepath=weights_file, monitor='val_loss', save_best_only=True))

history = model.fit_generator(train_generator, samples_per_epoch / batch_size, nb_epoch, callbacks=callbacks, verbose=2,
                              validation_data=val_generator, validation_steps=N_seq_val / batch_size)
# Save the model
if save_model:
    json_string = model.to_json()
    with open(json_file, "w") as f:
        f.write(json_string)
