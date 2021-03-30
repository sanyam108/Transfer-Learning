from keras.models import Model,Sequential

from keras.layers import Dense,Flatten,Activation,Dropout

from keras.applications.vgg16 import VGG16

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os

filenames = os.listdir("C:\\Users\\sanyam ahuja\\Documents\\Python\\Transfer learning\\train\\cat")
categories = []
for filename in filenames:
    categories.append('cat')
        
cat = pd.DataFrame({'filename': filenames,'category': categories})

filenames = os.listdir("C:\\Users\\sanyam ahuja\\Documents\\Python\\Transfer learning\\train\\dog")
categories = []
for filename in filenames:
    categories.append('dog')       
        
dog = pd.DataFrame({'filename': filenames,'category': categories})

train = pd.concat([cat,dog])

filenames = os.listdir("C:\\Users\\sanyam ahuja\\Documents\\Python\\Transfer learning\\test1")
categories = []
for filename in filenames:
    category = filename.split('.')[0]
    if category == 'dog':
        categories.append('dog')
    else:
        categories.append('cat')

test = pd.DataFrame({'filename': filenames,'category': categories})

train = train.reset_index()

train = train.drop(['index'],axis=1)

train

train.info()

test.info()

IMAGE_SIZE = [224,224]

vgg = VGG16(input_shape=IMAGE_SIZE + [3],include_top=False)

for layer in vgg.layers:
    layer.trainable = False

x = Flatten()(vgg.output)
prediction = Dense(2, activation='softmax')(x)

model = Model(inputs=vgg.input,outputs=prediction)

model.summary()

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)


training_set = train_datagen.flow_from_directory('C:\\Users\\sanyam ahuja\\Documents\\Python\\Transfer learning\\train',
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')


test_datagen = ImageDataGenerator(rescale=1./255)

test_set = test_datagen.flow_from_dataframe(
                                            test, 
                                            'C:\\Users\\sanyam ahuja\\Documents\\Python\\Transfer learning\\test1', 
                                            x_col='filename',
                                            y_col='category',
                                            target_size=IMAGE_SIZE,
                                            class_mode='categorical',
                                            batch_size=15
)

r = model.fit(training_set,
              validation_data=test_set,
              epochs=5,
              steps_per_epoch=len(training_set),
              validation_steps=len(test_set)
)

model_loss = pd.DataFrame(model.history.history)

model_loss

plt.figure(figsize=(12,8))
model_loss[['loss','val_loss']].plot()

model_loss[['accuracy','val_accuracy']].plot()

model.save('face_recognition.h5')

