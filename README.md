# predirep-paper

## Overview

PrediRep is a deep learning network inspired by predictive coding, a neuroscientific theory of information processing in the brain. This repository hosts the implementation and details of the PrediRep model. PrediRep is an improved model over other similar predictive    coding inspired deep learning models such as PredNet and PreCNet. It achieves closer functional and structural alignment with predictive coding while maintaining comparable computational performance.

## Features

PrediRep is composed of hierarchically stacked levels, each of which contains an R and an E layer. The operations of PrediRep can be split into a downward and an upward sweep. During the downward sweep, information flows from higher to lower R layers through feedback connections. The R layers employ local recurrent connections to update their activity based on the feedback signal before transmitting their own activity further down the hierarchy. Simultaneously, through a second feedback connection, the R layers generate predictions (Â), which are transmitted to the E layers of the level below. After the downward sweep, an upward sweep occurs. During this phase, the E layers calculate prediction errors and send their activity forward to the R layers of the level above. These R layers, in turn, update their activity again using local recurrent connections, this time based on the prediction errors. The updated activities of the R layers are then sent through lateral connections to the E layers of the same level in the form of targets (A). At the lowest level of the hierarchy, the target is the current input (I). Therefore, except for the lowest R layer, the R layers are engaged in predicting a mixture of their own past and current activities, refined by the input. 

## Setup

To quickly get started with PrediRep you first need to create a conda environment. The conda environment should be created from the environment.yml file, which allows for a quick way to set up the environment to run PrediRep. Help on how to do this can be found under https://conda.io/docs/user-guide/tasks/manage-environments.html. Once you have created the environment, you first need to go to the following directory: WHERE_YOUR_ANACONDA_IS_STORED/anaconda3/envs/predirep/lib/python3.6/site-packages/keras/engine. You then need to copy the savings.py file found in the setup directory of this repository to the aforementioned directory and overwrite the file currently found there. After this, you can run the scripts with the pre-trained PrediRep model.

## Citation
If you find PrediRep useful for your work, please consider citing our paper (will be released soon).

## References

Prednet - https://coxlab.github.io/prednet/

PreCNet - https://arxiv.org/abs/2004.14878

RBP - https://arxiv.org/abs/2005.03230

## Contact

ibrahimhashim.coding@gmail.com


