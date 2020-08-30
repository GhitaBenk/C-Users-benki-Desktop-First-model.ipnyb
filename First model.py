#!/usr/bin/env python
# coding: utf-8

# # Crime prediction in St Louis City

# In[2]:


import pandas as pd
import seaborn as sns
import datetime
import numpy as np
from datetime import datetime
from datetime import timedelta 
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import StandardScaler # for normalization
from sklearn.preprocessing import MinMaxScaler


# # Preprocessing data

# Run time may be long, you can directly load pre-processed data attached

# In[3]:


#importing dataset
crime=pd.read_csv('stl-crime-data_2008-2015.tsv',delimiter='\t',encoding='utf-8')
#keeping wanted columns
crime = crime.drop(columns=["Unnamed: 0","FileName","CADAddress","CADStreet","CodedMonth","Complaint","Count","Crime","ShortCrimeCode","UCRType","Description","District","FlagAdministrative","FlagCleanup","FlagCrime","FlagUnfounded","ILEADSAddress","ILEADSStreet","LocationComment","LocationName","Neighborhood","NeighborhoodPrimaryDistrict","NeighborhoodAddlDistrict","Year"])
#For simplification, focusing on one crime type

crime['Buckets']=['b' for i in range(len(crime.index))]
crime['DateOccured'] = [datetime.strptime(crime.iloc[i]['DateOccured'], '%m/%d/%Y %H:%M') for i in range(len(crime.index))]
print(crime)


# In[4]:


#keeping coordinates within city borders
filter1 = crime['Latitude'] > 38.5 
filter2 = crime['Latitude'] < 38.9
filter3 = crime['Longitude'] > -90.4 
filter4 = crime['Longitude'] < -90.1
filter5 = crime['UCRCrime'] == 'Larceny-theft' #For simplification, focusing on one crime type
crime = crime.where(filter1 & filter2 & filter3 & filter4 & filter5)
lat_max = crime['Latitude'].max()
lat_min = crime['Latitude'].min()
long_max = crime['Longitude'].max()
long_min = crime['Longitude'].min()
crime = crime.dropna()
print(crime)


# In[5]:


pas = 0.01
n_lat = int((lat_max-lat_min)/pas) + 1
n_long = int((long_max-long_min)/pas) + 1

#grouping coordinates in buckets (1bucket ~ 100 m²)
for i in range(0, n_lat): 
      for k in range(0, n_long):


            lat_inf = lat_min + pas*i
            lat_sup = lat_min + pas*(i+1)
            long_inf = long_min + pas*k
            long_sup = long_min + pas*(k+1)

            mask = (crime['Latitude'] > lat_inf) & (crime['Latitude'] < lat_sup) & (crime['Longitude'] > long_inf) & (crime['Longitude'] < long_sup)
            crime.loc[mask,'Buckets'] = 'bucket_'+str(i)+'_'+ str(k)


# In[108]:


allmonths = [] #list for each month
bucketsname = []
#contains matrices of count for each half day for each bucket
for month in range(12*8):
    data = pd.DataFrame({'buckets' : list(set(crime['Buckets']))})
    t = datetime.strptime('Jan 1 2008 00:00','%b %d %Y %H:%M')+ relativedelta(months=month)
    for k in range(60) :
        t = t +timedelta(hours= k*12)
        cond1 = (crime.DateOccured > t) & (crime.DateOccured< t+ timedelta(hours=12))
        d1 = crime.where(cond1)
        c = d1.groupby('Buckets')['Buckets'].count().to_dict()
        A = pd.DataFrame(list(c.items()),columns = ['buckets',k]) 
        D =data.merge(A ,how = 'outer' , on = 'buckets')
        data = D
    
    bucketsname.append(D.fillna(0))
    allmonths.append(D.fillna(0).drop(columns=['buckets']))


 


# In[7]:


import pickle
with open("allmonths.txt", "wb") as fp:   #Pickling
    pickle.dump(allmonths, fp)


# In[109]:


X = allmonths.copy()


# In[110]:


for i in range(len(X)):
    X[i] = np.array(X[i]).T
X= np.array(X)


# In[10]:


np.save( 'Dataset_crime_pred', X, allow_pickle = True)


# # Nombre de crime pour chaque mois en fonction des buckets

# In[11]:


import matplotlib.pyplot as plt 
A = []
B = []
for i in range(len(X)) :
    for j in range(len(X[i])):
        
        A += [np.sum((X[i].T)[j])]
        B += [(bucketsname[i])['buckets'][j]]
    plt.figure(figsize=(40,10))
    plt.scatter(B,A)    
    plt.show()


# # LSTM implementation

# In[125]:


X = np.load('Dataset_crime_pred.npy')
X


# In[170]:


#splitting data
x_train , y_train, x_test, y_test = X[:72] ,X[1:73],X[73:-1],X[74:]
#x_train_LSTM = np.reshape(x_train, (np.shape(x_train)[0],1, np.shape(x_train)[1]))
#x_test_LSTM = np.reshape(x_test, (np.shape(x_test)[0],1, np.shape(x_test)[1]))


# In[267]:


x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)
print(y_train.shape, x_test.shape)


# In[172]:


x_train_log = np.where(x_train < 1 ,x_train ,1)
y_train_log = np.where(y_train < 1 ,y_train ,1)
x_test_log = np.where(x_test < 1 ,x_test ,1)
y_test_log = np.where(y_test < 1 ,y_test ,1)


# In[173]:


print(y_test.shape, x_test.shape)


# In[174]:


from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers import BatchNormalization , ConvLSTM2D, Reshape, Flatten ,Bidirectional , RepeatVector

from keras.layers.core import Dense, Activation, Dropout
from sklearn.metrics import mean_squared_error
from keras.layers import Dropout
from keras import regularizers


# In[175]:


x_cnn = x_train.reshape(-1,60,20, 10,1 )
x_tcnn = x_test.reshape(-1,60,20, 10,1 )


# In[381]:


epochs = 80
batch_size = 20

model = Sequential() 
model.add(LSTM(200,recurrent_activation = 'relu', activation ='sigmoid',kernel_regularizer = regularizers.l2(0.1), input_shape = (60,200), return_sequences= True))
model.compile(loss = "mean_absolute_error", 
              optimizer = "adam"
                 )

model.summary()

history = model.fit(x_train,y_train, epochs=epochs, batch_size=batch_size,validation_data=(x_test, y_test), verbose=1)


# In[382]:


import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()


# In[383]:


yhat = model.predict(x_test)


# In[265]:


import warnings
warnings.filterwarnings('ignore')
labels = []
for i in range(172,175):
    plt.plot(range(60),yhat[11].T[i]*100, c = 'r')
    plt.plot(range(60),y_test[11].T[i],  marker='.')
    #labels.append('Bucket '+str(i))
    label = ['Bucket '+str(i)+' predicted', 'Bucket '+str(i)+' actual']
    plt.legend(label)
    plt.xlabel('demi journée')
    plt.ylabel('count')
    plt.show()


# In[150]:


plt.scatter(range(60),scaled.T[11]),scaled.T[11]


# In[261]:


from sklearn.metrics import accuracy_score

y_true = np.argmax(y_log[11], axis = 1)
y_pred = np.argmax(yhat_log[11], axis = 1)
accuracy_score(y_true, y_pred)


# In[262]:


yhat_log[11].shape , y_log[11].shape , y_test_log.shape


# In[78]:


from sklearn.preprocessing import MinMaxScaler
import pandas as pd

scaler = MinMaxScaler(feature_range=(0, 1)).
scaled = scaler.fit_transform(y_test[11])


# In[208]:


np.argmax(yhat[11],axis = 0) ,np.argmax(y_test[11],axis = 0)


# In[384]:


y_log = np.where(y_test.T < 1 ,y_test.T ,1)


# In[269]:


y_test[0].T.shape,  y_log[0]


# In[271]:


accuracy_score(y_log.T[10], yhat_log.T[10])


# In[270]:


yhat_log = np.where(yhat > 0.25 , 1 , 0)


# In[52]:


y_test[0].T.shape


# # Accuracyy
# 

# In[468]:


A = []
accuracy  = np.array([])
for thresh in np.arange(0,np.max(yhat),0.0005):
    T = []

    yhat_log = np.where(yhat.T > thresh , 1 , 0)
    for i in range(200):
        y_pred = yhat_log[i]
        y_true = y_log[i]
        T.append(accuracy_score(y_true, y_pred))
        
    accuracy = np.hstack((accuracy,np.array(T)))
    A.append(np.mean(T))
#plt.plot(range(12),T)
#plt.show()


# In[ ]:





# In[463]:


acc = np.array(accuracy)
thresh_ind = np.argmax(acc,axis =0)
np.argmax(thresh_ind)


# In[459]:


L = []
for i in range(21):
    L.append(np.mean(acc[i]))

print(L,L.index(max(L)))


# In[437]:


yhat.T.shape


# In[427]:


np.max(yhat), np.max((yhat.T)[0]) , np.arange(0,np.max(yhat),0.0005)[thresh_ind[0]]


# In[460]:


yhat_log = np.zeros((200,60,22))
for i in range(200):
    yhat_log[i] =  np.where((yhat.T)[i] > np.arange(0,np.max(yhat),0.0005)[thresh_ind[i]] , 1 , 0)
    
    
yhat_log    


# In[461]:


T = []
accuracy  = np.array([])
for i in range(200):
    y_pred = yhat_log[i]
    y_true = y_log[i]
    T.append(accuracy_score(y_true, y_pred))
accuracy = np.hstack((accuracy,np.array(T)))
plt.figure(figsize = (10,10))
plt.plot(range(200),accuracy)
plt.show()


# In[464]:


import warnings
warnings.filterwarnings('ignore')
labels = []
for i in range(200):
    y_pred = yhat_log[i]
    y_true = y_log[i]
    print(accuracy_score(y_true,y_pred))
    plt.plot(range(60),y_pred.T[11], c = 'r')
    plt.scatter(range(60),y_true.T[11])
    #labels.append('Bucket '+str(i))
    label = ['Bucket '+str(i)+' predicted', 'Bucket '+str(i)+' actual']
    plt.legend(label)
    plt.xlabel('demi journée')
    plt.ylabel('count')
    plt.show()


# In[339]:


y_pred.shape


# In[475]:


acc = accuracy.reshape(200,21)
np.arange(0,np.max(yhat),0.005).shape


# In[480]:


for i in range(1):
    plt.figure(figsize = (10,5))
    plt.plot(np.arange(0,0.105,0.005),acc[i])
    plt.xlabel('seuil')
    plt.ylabel('Précision')
    plt.show()


# In[128]:


y_pred = np.zeros((60,12))
y_true = y_log[1]

accuracy_score(y_true, y_pred)


# In[125]:


y_log[1].shape


# In[161]:


np.arange(0,1,0.01)[np.argmax(acc, axis = 1)]


# In[79]:


a = y_test.T[0].T
a = np.where(a < 1 , 0,1) 


# In[85]:


np.arange(0.5,0.8,0.1)



# # Résultat LSTM

# In[160]:


mois = 11
from sklearn.metrics import classification_report
y_true = np.argmax(y_test[mois], axis = 1)
y_pred = np.argmax(yhat[mois], axis = 1)
print(classification_report(y_true, y_pred))


# # LSTM BIDIRECTIONAL

# In[361]:


epochs = 80
batch_size = 20

model1 = Sequential() 
model1.add(Bidirectional(LSTM(100,recurrent_activation = 'relu', activation ='sigmoid',kernel_regularizer = regularizers.l2(0.1), input_shape = (60,200))))
model1.add(RepeatVector(60))
model1.add(Bidirectional(LSTM(100, activation='relu', return_sequences=True )))

model1.compile(loss = "mean_absolute_error", 
              optimizer = "adam"
                 )


history = model1.fit(x_train,y_train, epochs=epochs, batch_size=batch_size,validation_data=(x_test, y_test), verbose=1 )


# In[188]:


import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()


# In[362]:


yhat = model1.predict(x_test)


# In[190]:


import warnings
warnings.filterwarnings('ignore')
labels = []
for i in range(109,113):
    plt.plot(range(60),yhat[11].T[i]*100, c = 'r')
    plt.scatter(range(60),y_test[11].T[i])
    #labels.append('Bucket '+str(i))
    label = ['Bucket '+str(i)+' predicted', 'Bucket '+str(i)+' actual']
    plt.legend(label)
    plt.show()


# In[363]:


y_log = np.where(y_test.T < 1 ,y_test.T ,1)


# In[370]:


A = []
accuracy  = np.array([])
for thresh in np.arange(0,1,0.005):
    T = []

    yhat_log = np.where(yhat.T > thresh , 1 , 0)
    for i in range(200):
        y_pred = yhat_log[i]
        y_true = y_log[i]
        T.append(accuracy_score(y_true, y_pred))
    accuracy = np.hstack((accuracy,np.array(T)))
    A.append(np.mean(T))
#plt.plot(range(12),T)
#plt.show()


# In[371]:


np.max(yhat), np.max(y_test)


# In[365]:


L = []
for i in range(29):
    L.append(np.mean(acc[i]))

print(L,L.index(max(L)))


# In[ ]:


acc = accuracy.reshape(200,200)


# # Résultat Bidirectional LSTM

# In[191]:


y_true = np.argmax(y_test[11], axis = 1)
y_pred = np.argmax(yhat[11], axis = 1)
accuracy_score(y_true, y_pred)


# In[249]:


from sklearn.metrics import classification_report
y_true = np.argmax(y_test[9], axis = 1)
y_pred = np.argmax(yhat[9], axis = 1)
print(classification_report(y_true, y_pred))


# In[46]:


n_mois = 20


# In[57]:


scaler = MinMaxScaler(feature_range=(0, 2))
scaled = scaler.fit_transform(x_test[n_mois].T)
# make a prediction
yhat = model1.predict(x_test)
#x_test = x_test.reshape((x_test.shape[0],[] 60))
# invert scaling for forecast
inv_yhat = np.concatenate((yhat[n_mois], x_test[n_mois]), axis=1)
inv_yhat = scaler.fit_transform(inv_yhat)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,:200]


# In[58]:



# invert scaling for actual
#y_test1 = y_test.reshape((len(y_test), 60))
inv_y = np.concatenate((y_test[n_mois], x_test[n_mois]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,:200]


# # Autoregression

# In[454]:


scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(fixed_bucket)
# make a prediction
yhat = model_regress1.predict(x_test)
x_test = x_test.reshape((x_test.shape[0], 60))
# invert scaling for forecast
inv_yhat = np.concatenate((yhat, x_test[:, -59:]), axis=1)
inv_yhat = scaler.fit_transform(inv_yhat)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]
# invert scaling for actual
y_test = y_test.reshape((len(y_test), 60))
inv_y = np.concatenate((y_test, x_test[:, -59:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]
aa=[x for x in range(len(inv_y))]
plt.plot(aa,inv_y, marker='.', label="actual")
plt.plot(aa, inv_yhat, 'r', label="prediction")
plt.ylabel('count', size=15)
plt.xlabel('Time step', size=15)
plt.legend(fontsize=15)
plt.show()


# # Lstm for every bucket 

# In[ ]:


x_train.reshape()


# In[ ]:


for i in range(len(x_train)):
    epochs = 80
    batch_size = 20

    

    model = Sequential() 
    model.add(LSTM(200,recurrent_activation = 'relu', activation ='sigmoid',kernel_regularizer = regularizers.l2(0.1), input_shape = (60,200), return_sequences= True))

    model.compile(loss = "mean_absolute_error", 
                  optimizer = "adam"
                     )

    model.summary()

    history = model.fit(x_train[i],y_train_log[i], epochs=epochs, batch_size=batch_size,validation_data=(x_test_log, y_test_log), verbose=1)

