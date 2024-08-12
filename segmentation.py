import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns
sns.set(style="whitegrid")
import os
import glob as gb
import cv2
import tensorflow as tf
import keras
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, label_ranking_average_precision_score, label_ranking_loss, coverage_error

from google.colab import drive
drive.mount('/content/drive')

imges_path = '/content/drive/MyDrive/tuft dental database/Radiographs/Radiographs/'
labels_path = '/content/drive/MyDrive/tuft dental database/Segmentation/Segmentation/teeth_mask/'

imge_path =  os.listdir(imges_path)
label_path = os.listdir(imges_path)

image_path_test = imge_path[:]
label_path_test = label_path[:]

image_path_test[2]

label_path_test[2]

X = []
image_name = []
for file in image_path_test : 
        image = plt.imread(imges_path+file)
        image = cv2.resize(image, (256,256))/255
        image = image.astype(np.float32)
        X.append(image)
        image_name.append(file)

image_name[2]

y = []
label_name = []
for file in label_path_test : 
        label = plt.imread(labels_path+ file.lower()).astype(np.float32)
        label = cv2.resize(label, (256,256))/255
        label = label.astype(np.float32)
        y.append(label)
        label_name.append(file)

label_name[2]

plt.subplot(1,2,1)
plt.imshow(X[3])
plt.title('Image')


plt.subplot(1,2,2)
plt.imshow(y[3])
plt.title('Label')

X = np.array(X)
y = np.array(y)

from sklearn.model_selection import train_test_split

X_train,X_test1,y_train,y_test1 = train_test_split(X,y ,test_size = 0.3 ,random_state=100)

print(X_train.shape)
print(y_train.shape)
print("***********")
print(X_test1.shape)
print(y_test1.shape)

X_test,X_validate,y_test,y_validate = train_test_split(X_test1,y_test1 ,test_size = 0.5 ,random_state=100)

print(X_test.shape)
print(y_test.shape)
print("***********")
print(X_validate.shape)
print(y_validate.shape)

from tensorflow.keras import Input
from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras.layers import Input, Activation, BatchNormalization, Dropout, Lambda, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
#K.set_image_data_format('channels_last')

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, UpSampling2D, Input, MaxPooling2D, Concatenate , Dropout
from tensorflow.keras.models import Model
from tensorflow import keras

def conv_block(x,num_filters):
    
    x=Conv2D(num_filters,(5,5),padding='same')(x)
    x=BatchNormalization()(x)
    x=Activation('relu')(x)
    
    x=Conv2D(num_filters,(3,3),padding='same')(x)
    x=BatchNormalization()(x)
    x=Activation('relu')(x)
    
    x=Conv2D(num_filters,(3,3),padding='same')(x)
    x=BatchNormalization()(x)
    x=Activation('relu')(x)
    x = Dropout(0.2)(x)
    
    
    return x
    

def build_model():
    size=256
    num_filters=[16,32,48,64]
    
    inputs=Input(shape=(size,size,3))
    skip_x=[]
    x=inputs
    
    #Encoder
    for f in num_filters:
        x=conv_block(x,f)
        skip_x.append(x)
        x=MaxPooling2D(2,2)(x)
        

    #bottleneck
    x=conv_block(x,num_filters[-1])
    
    num_filters.reverse()
    skip_x.reverse()
    
    #Decoder
    for i,f in enumerate(num_filters):
        x=UpSampling2D((2,2))(x)
        xs=skip_x[i]
        x=Concatenate()([x,xs])
        x=conv_block(x,f)
    #output
    x=Conv2D(3,(1,1),padding='same')(x)
    x=Activation('sigmoid')(x)
    return Model(inputs,x)

model = build_model()
model.summary()

import os
import numpy as np
import cv2
from glob import glob
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard

def iou(y_true, y_pred):
    def f(y_true, y_pred):
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        x = (intersection + 1e-15) / (union + 1e-15)
        x = x.astype(np.float32)
        return x
    return tf.numpy_function(f, [y_true, y_pred], tf.float32)

from tensorflow.keras import backend as K
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 0.0001) / (K.sum(y_true_f) + K.sum(y_pred_f) + 0.0001)

def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

pip install tensorflow_addons

import tensorflow_addons as tfa
step = tf.Variable(0, trainable=False)

schedule = tf.optimizers.schedules.PiecewiseConstantDecay([10000, 15000], [1e-0, 1e-1, 1e-2])

lr = 1e-1 * schedule(step)
wd = lambda: 1e-4 * schedule(step)

optimizer = tfa.optimizers.AdamW(learning_rate=lr, weight_decay=wd)

model.compile(optimizer=optimizer, loss=[dice_coef_loss], metrics=[dice_coef , iou])

callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
history = model.fit(X_train,y_train,epochs=100, validation_data=(X_validate , y_validate), callbacks=[callback])

# converting hsitory to dataframe
pd.DataFrame(history.history)

pd.DataFrame(history.history)[['dice_coef', 'val_dice_coef']].plot()
plt.title('Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('accuracy')

pd.DataFrame(history.history)[['loss', 'val_loss']].plot()
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')

pd.DataFrame(history.history)[['iou', 'val_iou']].plot()
plt.title('Model Performance')
plt.xlabel('Epochs')
plt.ylabel('iou')

y_pred = model.predict(X_test)

print("Evaluate on test data")
results = model.evaluate(X_test, y_test)
print("test loss, test acc:", results)

import cv2
cv2.imwrite('color_img.jpg', X_test[0])
img = cv2.imread("./color_img.jpg", cv2.IMREAD_COLOR)
plt.imshow(img)

plt.subplot(1,3,1)
plt.imshow(img)
plt.title('Real Image')

plt.subplot(1,3,2)
plt.imshow(y_test[0])
plt.title('Real Label')


plt.subplot(1,3,3)
plt.imshow(y_pred[0])
plt.title('Predicted Label')
