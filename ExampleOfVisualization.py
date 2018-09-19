# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 11:32:05 2018

@author: whe
"""

from keras.applications import ResNet50
from vis.utils import utils
from keras import activations

model = ResNet50(weights='imagenet', include_top= True)

model.summary()

layer_idx = utils.find_layer_idx(model)