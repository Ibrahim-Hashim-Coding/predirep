'''
Visualise layer specific unit representations for PrediRep on dataset.
'''
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import matplotlib.pyplot as plt
from data_settings import *

# Load parameters
units = ['R', 'Ahat', 'A', 'E']  # which units to visualise
nt = 10  # which time step to visualise units from
layers = [0, 1, 2, 3]  # which layers to visualise units from

units_and_layers = ["{}{}".format(unit, layer) for unit in units for layer in layers]

# Create figures
fig = plt.figure(figsize=(len(layers), len(units)))
ax = [plt.subplot(len(units), len(layers), i + 1) for i in range(len(layers) * len(units))]
plt.subplots_adjust(wspace=0.1, hspace=-0.4)

# Remove ticks
for a in ax:
    a.set_xticks([])
    a.set_yticks([])

for unl_c, unl in enumerate(units_and_layers):
    unit_rep = np.load(WEIGHTS_DIR + 'unit_rep/{}.npy'.format(unl))
    unit_rep = unit_rep[0, nt-1]  # only extracted wanted time step

    # In this situation, it would be good to visualise Ahat0 and A0 in colour, as they have 3 colour channels,
    # and the rest as grayscale images as they have more than 3 colour channels. In the current implementation, Ahat0
    # is the 5th element, and A0 is the 9th element.
    if unl_c == 4 or unl_c == 8:
        ax[unl_c].imshow(unit_rep)  # keep dimensiosn for coloured image
    else:
        ax[unl_c].imshow(np.mean(unit_rep, axis=2))  # average over dimensions to create grayscale image


# Create labels for this situation
ax[0].set_ylabel('R')
ax[len(layers)].set_ylabel(r'$\hat{A}$')
ax[len(layers)*2].set_ylabel('A')
ax[len(layers)*3].set_ylabel('E')

ax[len(layers)*3].set_xlabel('Layer 0')
ax[len(layers)*3+1].set_xlabel('Layer 1')
ax[len(layers)*3+2].set_xlabel('Layer 2')
ax[len(layers)*3+3].set_xlabel('Layer 3')

plt.savefig(WEIGHTS_DIR + 'unit_representations.png', bbox_inches='tight')
plt.close()
