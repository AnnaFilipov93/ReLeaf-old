
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Activation, Dropout,Flatten,Conv2D,MaxPool2D,Dense
from keras.preprocessing import image


# In[2]:


#preparing the data
image_gen=ImageDataGenerator(rotation_range=30, width_shift_range=0.1, 
                            height_shift_range=0.1, rescale=1/255, shear_range=0.2,
                            zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')

image_shape=(240,240,3)


# In[3]:


##~~building the model~~##


model=Sequential()
#convolutional layer_I
model.add(Conv2D(filters=32,kernel_size=(3,3),input_shape=image_shape,activation='relu'))
#pool layer_I
model.add(MaxPool2D(pool_size=(2,2)))

#convolutional layer_II
model.add(Conv2D(filters=64,kernel_size=(3,3),input_shape=image_shape,activation='relu'))
#pool layer_II
model.add(MaxPool2D(pool_size=(2,2)))

#convolutional layer_III
model.add(Conv2D(filters=64,kernel_size=(3,3),input_shape=image_shape,activation='relu'))
#pool layer_III
model.add(MaxPool2D(pool_size=(2,2)))

#flatten
model.add(Flatten())

#dense layer
model.add(Dense(128,activation='relu'))

#dropout layer
model.add(Dropout(0.5))

#output layer
model.add(Dense(1,activation='sigmoid'))

#compile
model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])



# In[4]:


##~~reading the data~~##

train_img=image_gen.flow_from_directory('FLOWERS\\train',
                                         target_size=image_shape[:2], batch_size=16, class_mode='binary')

test_img=image_gen.flow_from_directory('FLOWERS\\test',
                                         target_size=image_shape[:2], batch_size=16, class_mode='binary')

#train_img.class_indices


# In[5]:


##~~fitting the model~~##

results=model.fit_generator(train_img, epochs=50,steps_per_epoch=75, validation_data=test_img)


# In[35]:


path='C://Users//Orit//Desktop//Computer-Vision-with-Python//FLOWERS//test//sunflower/22416421196_caf131c9fa_m.jpg'
flower=cv2.cvtColor(flower,cv2.COLOR_BGR2RGB)
plt.imshow(flower)

flower_file=path
flower_img=image.load_img(flower_file, target_size=(240,240))
flower_img=image.img_to_array(flower_img)
flower_img=np.expand_dims(flower_img,axis=0)
flower_img=flower_img/255


# In[36]:


p=model.predict_classes(flower_img)

pred(p)


# In[30]:


def pred (p):
    if p==0:
        print("the model recognized it as a rose")
    elif p==1:
         print("the model recognized it as a sunflower")


# In[37]:


model.save('rose_sunflower_model.h5')

