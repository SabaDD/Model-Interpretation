import numpy as np
import os
import time
#import sys
#sys.path.append('/opt/packages/keras/keras_2.0.4/kerasEnv/lib/python2.7/site-packages/')

#from vgg16 import VGG16
from resnet50 import ResNet50
from keras.preprocessing import image
from imagenet_utils import preprocess_input, decode_predictions
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers import merge, Input
from keras.models import Model
from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, StratifiedKFold
from keras import optimizers
import matplotlib.pyplot as plt
from keras.wrappers.scikit_learn import KerasClassifier
from keras.preprocessing.image import ImageDataGenerator

import  tensorflow as tf
from sklearn.model_selection import StratifiedKFold

from prediction import model_prediction


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


PATH = os.getcwd()
data_path = PATH + '/Data_case_control_prior1'
data_dir_list = os.listdir(data_path)

img_data_list = []

for dataset in data_dir_list:
    img_list = os.listdir(data_path +'/' +dataset)
    print('loaded the images from dataset-'+'{}\n'.format(dataset))
    for img in img_list:
        splt_img = img.split("_")
        img_path = data_path + '/' + dataset + '/' + img
        img = image.load_img(img_path, target_size= (224,224))
        if(splt_img[1][0] == 'R'):
            img = image.image.flip_axis(img,1)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        print('Input image shape', x.shape)
        img_data_list.append(x)


img_data = np.array(img_data_list)
print(img_data.shape)
img_data = np.rollaxis(img_data, 1,0)
print(img_data.shape)
img_data = img_data[0]
print(img_data.shape)
# mean of the images
##img_mean
img_data -= np.mean(img_data)
img_data /= np.std(img_data)


# define label
num_classes = 2
num_of_samples = img_data.shape[0]
labels = np.ones((num_of_samples,), dtype='int64')


labels[0:113] = 1
labels[113:] = 0
print("the labels are: {}".format(labels))
names = ['Cancer', 'Control']

Y = np_utils.to_categorical(labels,num_classes)

#x,y = shuffle(img_data, Y)
#x,y = shuffle(img_data, Y, random_state = 2)


#X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.20, random_state= 2)
X_train, X_test, y_train, y_test = train_test_split(img_data,Y, test_size=0.20, random_state= 2)

# define the ResNet model
image_input = Input(shape=(224,224,3))
model = ResNet50(input_tensor=image_input, include_top=True, weights='imagenet')
model.summary()

last_layer = model.get_layer('avg_pool').output
x = Flatten(name='flatten')(last_layer)
out = Dense(num_classes, name='output_layer', activation='softmax')(x)
#out = Dropout(0.5)(out)
custom_resnet_model = Model(inputs= image_input, outputs=out)
custom_resnet_model.summary()

for layer in custom_resnet_model.layers[:-2]:
    layer.trainable = False


opt = optimizers.SGD(lr= 0.001, momentum= 0.8, decay= 1e-6, nesterov= False)
#opt= optimizers.RMSprop(lr= 0.0001, rho= 0.9, decay= 0, epsilon= None)
#opt = optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)
#opt=optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
#opt = optimizers.Adam(lr=1e-6, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-4, amsgrad=False)
#opt =optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)

custom_resnet_model.compile(loss='categorical_crossentropy', optimizer=opt,metrics=['accuracy'])
#custom_resnet_model.compile(loss='binary_crossentropy', optimizer=opt,metrics=['accuracy'])


seed = 7
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
cvscores = []
ii = 0
f_acc_eval = open('accForEvalModel.txt','w')
for train_indices, valid_indices in kfold.split(X_train,y_train[:,1]):
    # Fit the model
    hist = custom_resnet_model.fit(X_train[train_indices], y_train[train_indices], epochs=20, batch_size=4, verbose=1)
    # Evaluate the model 
    scores = custom_resnet_model.evaluate(X_train[valid_indices], y_train[valid_indices], verbose=1)
    print("%s: %.2f%%" % (custom_resnet_model.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)
    
    
    print("Average Accuracy = %.2f%%"%np.mean(cvscores))
    print("Std = %.2f%%"%np.std(cvscores))
    f_acc_eval.write("%s " % np.mean(cvscores))
    f_acc_eval.write("%s \n" % np.std(cvscores))
    f_auc = open('aucForEachSlidingWindow_'+str(ii)+'.txt', 'w')
    for left in range(0,169,56):
        for upper in range(0,169,56):
         auc_sliding_window = model_prediction(custom_resnet_model, X_train, y_train, left, left+56, upper, upper+56)
         f_auc.write("%s\n" % auc_sliding_window)
    f_auc.close()
    ii= ii+1

f_acc_eval.close()







