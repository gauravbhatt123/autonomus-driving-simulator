import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
#
from utils import *

path = 'myData'
data = importDataInfo(path)
data=balancedata(data,display=False)

imagespath,steering=loaddata(path,data)
print(imagespath[0],steering[0])

xtrain,xval,ytrain,yval= train_test_split(imagespath,steering,test_size=0.2,random_state=5)
print('TOTAL TRAINING IMAGES :',len(xtrain))
print('TOTAL TESTING IMAGES :',len(xval))


model=createmodel()
model.summary()

history=model.fit(batchgenerator(xtrain,ytrain,100,1),steps_per_epoch=300,epochs=10,validation_data=batchgenerator(xval,yval,100,0),validation_steps=200)
model.save('model.h5')
print('Model saved')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training','Validation'])
plt.ylim([0,1])
plt.title('Loss')
plt.xlabel('Epoch')
plt.show()