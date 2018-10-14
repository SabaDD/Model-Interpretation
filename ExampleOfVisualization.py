# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 11:32:05 2018

@author: whe
"""

from keras.applications import ResNet50
from vis.utils import utils
from keras import activations
from medpy.io import load
import SimpleITK as sitk
import os

#model = ResNet50(weights='imagenet', include_top= True)
#
#model.summary()
#
#layer_idx = utils.find_layer_idx(model)
#Read image using simpleITK
PATH = os.getcwd()
itkimage = sitk.ReadImage(PATH+'\BARP0807\LCC.mhd')

#Convert the image into numpy array and then shuffle the dimentions to get axis in the order z, y, x
mamograph = sitk.GetArrayFromImage(itkimage)
