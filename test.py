#!/usr/bin/env python
# coding: utf-8

# In[4]:


from keras.preprocessing.sequence import pad_sequences
from keras.layers import LSTM,Bidirectional,Dense,Input,Conv2D,MaxPool2D,Lambda,BatchNormalization,Reshape,Lambda
from keras.models import Model
import keras.backend as K
import cv2


# In[6]:


inputs = Input(shape=(32,128,1))
 

conv1 = Conv2D(64, (3,3), activation = 'relu', padding='same')(inputs)
pool1 = MaxPool2D(pool_size=(2, 2), strides=2)(conv1)
conv2 = Conv2D(128, (3,3), activation = 'relu', padding='same')(pool1)
pool2 = MaxPool2D(pool_size=(2, 2), strides=2)(conv2)
conv3 = Conv2D(256, (3,3), activation = 'relu', padding='same')(pool2)
conv4 = Conv2D(256, (3,3), activation = 'relu', padding='same')(conv3)
pool4 = MaxPool2D(pool_size=(2, 1))(conv4)
conv5 = Conv2D(512, (3,3), activation = 'relu', padding='same')(pool4)
batchnorm5 = BatchNormalization()(conv5)
conv6 = Conv2D(512, (3,3), activation = 'relu', padding='same')(batchnorm5)
batchnorm6 = BatchNormalization()(conv6)
pool6 = MaxPool2D(pool_size=(2, 1))(batchnorm6)
conv7 = Conv2D(512, (2,2), activation = 'relu')(pool6)
squeezed = Lambda(lambda x: K.squeeze(x, 1))(conv7)
blstm1 = Bidirectional(LSTM(128, return_sequences=True, dropout = 0.2))(squeezed)
blstm2 = Bidirectional(LSTM(128, return_sequences=True, dropout = 0.2))(blstm1)
outputs = Dense(63, activation = 'softmax')(blstm2)
test_model = Model(inputs, outputs)


# In[30]:


image=cv2.imread("C:/Users/joyje/Pictures/p1.png",0)


# In[31]:


import numpy as np
from scipy import misc
test=[]


image=np.expand_dims(image,axis=2)
image=image/255
test.append(image)


# In[32]:


test=np.array(test)


# In[33]:


import string
char_list=string.ascii_letters+string.digits
test_model.load_weights('bestmodel')
prediction = test_model.predict(test)
out = K.get_value(K.ctc_decode(prediction, input_length=np.ones(prediction.shape[0])*prediction.shape[1],
                         greedy=True)[0][0])
i = 0
for x in out:
    
    print("predicted text = ", end = '')
    for p in x:  
        if int(p) != -1:
            print(char_list[int(p)], end = '')       
    print('\n')
    i+=1


# In[ ]:




