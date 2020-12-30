import os 
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.layers import (Dense , Dropout ,Flatten ,
                                     MaxPool2D ,Conv2D ,Activation ,
                                     LeakyReLU)
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.activations import softmax ,relu
from tensorflow.keras.preprocessing.image import load_img , img_to_array 
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.activations import relu ,softmax
process_dir ="C:\\Datasets\\chest_xray\\chest_xray"

test_dir = process_dir + "\\test"
train_dir = process_dir + "\\train"
validation_dir = process_dir + "\\validation"

to_plot = []

train_data_NORMAL = []
test_data_NORMAL = []

train_data_PNEUMONIA = []
test_data_PNEUMONIA = []

index= 0 
for img_path_test1 , img_path_test2 in\
    zip(os.listdir(test_dir + "\\NORMAL")  , os.listdir(test_dir + "\\PNEUMONIA")):
    index+=1
    verbose = ("normal  & pneumonia test  image " + str(index) +" / " + str(os.listdir(test_dir + "\\NORMAL").__len__()))
    print(verbose)
    test_image1 =  img_to_array(load_img(test_dir + '\\NORMAL\\' +img_path_test1 ,
                              target_size = (130 ,130,1)  ,
                              color_mode=  "grayscale"))
    test_image2 =  img_to_array(load_img(test_dir + '\\PNEUMONIA\\' +img_path_test2 ,
                              target_size = (130 ,130,1) ,
                              color_mode = "grayscale"))
    test_data_NORMAL.append((test_image1))
    test_data_PNEUMONIA.append((test_image2))
            
    
    to_plot.append((test_image1.shape))

print()
index= 0
for img_path_train in os.listdir(train_dir +"\\NORMAL"):
    try:
        train_image  = img_to_array(load_img(train_dir + '\\NORMAL\\' +img_path_train ,
                                  target_size = (130,130,1) ,
                                  color_mode = "grayscale"))
        train_data_NORMAL.append((train_image))
    except:
        pass
    index+=1
    verbose =("normal train image " + str(index) + " / " + str(os.listdir(train_dir +"\\NORMAL").__len__()))
    print(verbose)


print()
index  = 0
for img_path_train in os.listdir(train_dir +"\\PNEUMONIA"):
    try:
        train_image  =  img_to_array(load_img(train_dir + '\\PNEUMONIA\\' +img_path_train ,
                                  target_size =(130 ,130 ,1) ,
                                  color_mode ="grayscale"))
        train_data_PNEUMONIA.append((train_image))
    except:
        pass
    index+=1
    verbose = ("Pneumonia train image " + str(index) + " / " + str(os.listdir(train_dir +"\\PNEUMONIA").__len__()))
    print(verbose)


#test images size plotted with seaborn
to_plot= np.array(to_plot)
sns.jointplot(to_plot[: ,0] , to_plot[: ,1])


test_data_NORMAL =  np.array(test_data_NORMAL)
train_data_NORMAL =  np.array(train_data_NORMAL )
test_data_PNEUMONIA =  np.array(test_data_PNEUMONIA)
train_data_PNEUMONIA =  np.array(train_data_PNEUMONIA )



test_data_in  = np.append(test_data_NORMAL ,test_data_PNEUMONIA ,axis = 0)
train_data_in= np.append(train_data_NORMAL , train_data_PNEUMONIA ,axis =0 )


train_data_out = np.ndarray(shape = (train_data_NORMAL.shape[0] + train_data_PNEUMONIA.shape[0] ,1) ,
                                   dtype = int)


train_data_out[:train_data_NORMAL.shape[0] , 0] = 0
train_data_out[train_data_NORMAL.shape[0]: , 0] = 1

test_data_out = np.ndarray(shape = (test_data_NORMAL.shape[0] + test_data_PNEUMONIA.shape[0] ,1) ,
                           dtype =int)

test_data_out[:test_data_NORMAL.shape[0] ,0] = 0
test_data_out[test_data_NORMAL.shape[0]: , 0]  =1


x_train  ,y_train ,x_test ,y_test = train_data_in , train_data_out ,\
                                    test_data_in ,test_data_out

x_train =  x_train/255
x_test = x_test/255

y_categorical_train = to_categorical(y_train)
y_categorical_test = to_categorical(y_test)

model =  Sequential()
model.add(Conv2D(filters = 32 ,kernel_size= (2 ,2) ,input_shape = (130 ,130 , 1) ,
                 strides = (1 ,1 ) ,padding ="same"))
model.add(Activation(relu))
model.add(MaxPool2D(pool_size = (2 ,2 ) , strides =(1  ,1) ,padding = "same"))
model.add(Conv2D(filters = 16 ,kernel_size= (2 ,2)  ,
                 strides = (1 ,1 ) ,padding ="same"))
model.add(Activation(relu))
model.add(MaxPool2D(pool_size = (2 ,2 ) , strides =(1  ,1) ,padding = "same"))

 
model.add(Flatten())

model.add(Dense(128)) 
model.add(Dropout(.5))
model.add(Activation(relu))
model.add(Dense(2))
model.add(Activation(softmax))

model.compile(optimizer  = "adam" ,loss = "categorical_crossentropy" , metrics =["accuracy"])


model.fit(x = x_train ,y = y_categorical_train , epochs  = 20, 
          verbose = 1 ,batch_size = 32 ,
          validation_data = (x_test ,y_categorical_test) ,
           )

model.save("pneumonia_model.h5")