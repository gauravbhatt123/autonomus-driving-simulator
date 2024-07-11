import cv2
import pandas as pd
import numpy as np
import os

from matplotlib import pyplot as plt
from sklearn.utils import shuffle
import matplotlib.image as mpimg
from imgaug import augmenters as iaa
import random
from tensorflow.keras import Sequential 
from tensorflow.keras.layers import Conv2D,Flatten,Dense,Input
from tensorflow.keras.optimizers import Adam



def getName(filepath):
    return filepath.split('\\')[-1]
def importDataInfo(path):

    columns= ['Center','Left','Right','Steering','Throttle','none','Speed']
    data=pd.read_csv(os.path.join(path,'driving_log.csv'),names=columns)
    print(data['Center'].iloc[0])
    # print(getName(data['Center'].iloc[0]))
    data['Center']=data['Center'].apply(getName)
    print(data['Center'].iloc[0])
    data['Left']=data['Left'].apply(getName)
    data['Right']=data['Right'].apply(getName)
    # print(data.head())
    return data


def balancedata(data,display=True):
     nbins=31
     samplesperbin=500
     hist,bins=np.histogram(data['Steering'],nbins)
     print(bins)
     if display:
         center=(bins[:-1]+bins[1:])*0.5
         print(center)
         plt.bar(center,hist,width=0.06)
         plt.plot((-1,1),(samplesperbin,samplesperbin))
         plt.show();
     removeindexlist=[]
     for j in range(nbins):
         bindatalist=[]
         for i in range(len(data['Steering'])):
             if data['Steering'][i]>=bins[j] and data['Steering'][i]<=bins[j+1]:
                 bindatalist.append(i)

         bindatalist = (shuffle(bindatalist))
         bindatalist=bindatalist[samplesperbin:]
         removeindexlist.extend(bindatalist)

     data.drop(data.index[removeindexlist],inplace=True)
     print('Remaining images',len(data))

     if display:
         hist, bins = np.histogram(data['Steering'], nbins)

         center = (bins[:-1] + bins[1:]) * 0.5
         # print(center)
         plt.bar(center, hist, width=0.06)
         plt.plot((-1, 1), (samplesperbin, samplesperbin))
         plt.show();

     return data

def loaddata(path,data):
    imagespath=[]
    steering=[]
    for i in range(len(data)):
        indexdata=data.iloc[i]
        # print(indexdata)
        imagespath.append(os.path.join(path,'IMG',indexdata.iloc[0]))
        print(os.path.join(path,'IMG',indexdata.iloc[0]))
        steering.append(float(indexdata.iloc[3]))


    imagespath=np.asarray(imagespath)
    steering=np.asarray(steering)
    return imagespath,steering


def augmentimage(imgpath,steering):

    img=mpimg.imread(imgpath)
    if np.random.rand() < 0.5:
        pan=iaa.Affine(translate_percent={'x':(-0.1,0.1),'y':(-0.1,0.1)})
        img=pan.augment_image(img)
    if np.random.rand() < 0.5:
        zoom=iaa.Affine(scale=(1,1.2))
        img=zoom.augment_image(img)
    if np.random.rand() < 0.5:
        brightness=iaa.Multiply((0.4,0.9))
        img=brightness.augment_image(img)
    if np.random.rand() < 0.5:
        img=cv2.flip(img,1)
        steering=-steering

    return img,steering

# imgre,st=augmentimage('center_2024_07_04_13_12_07_815.jpg',0)
# plt.imshow(imgre)
# plt.show()

def preprocessing(img):
    img=img[60:135,:,:]
    img=cv2.cvtColor(img,cv2.COLOR_RGB2YUV)
    img=cv2.GaussianBlur(img,(3,3),0)
    img=cv2.resize(img,(200,66))
    img=img/255
    return img
imgre=preprocessing(mpimg.imread('center_2024_07_04_13_12_07_815.jpg'))
plt.imshow(imgre)
plt.show()


def batchgenerator(imagepath,steeringlist,batchsize,trainflag):
    while True:
        imgbatchlist=[]
        steeringbatchlist=[]
        for i in range(batchsize):
            index = random.randint(0, len(imagepath) - 1)
            if trainflag:
                img,steering=augmentimage(imagepath[index],steeringlist[index])
            else:
                img=mpimg.imread((imagepath[index]))
                steering=steeringlist[index]

            img=preprocessing(img)
            imgbatchlist.append(img)
            steeringbatchlist.append(steering)
        yield(np.asarray(imgbatchlist),np.asarray(steeringbatchlist))


def createmodel():
    model = Sequential()
    model.add(Input(shape=(66, 200, 3)))
    model.add(Conv2D(24, (5, 5), (2, 2), activation='elu'))
    model.add(Conv2D(36, (5, 5), (2, 2), activation='elu'))
    model.add(Conv2D(48, (5, 5), (2, 2), activation='elu'))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))

    model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse')
    return model




