from numpy.core.fromnumeric import shape
import pandas as pd
import numpy as np
import tensorflow as tf
import os,sys
from sklearn.model_selection import train_test_split

PRICING = pd.read_csv(r'C:\\Users\\ethan\\OneDrive - University of Tennessee\\Fall_2021\\BZAN544-Ballings\\pricing_final.csv')
#PRICING.sku = pd.Categorical(PRICING.sku);PRICING.category = pd.Categorical(PRICING.category)

#X's
sku = PRICING['sku'].to_numpy()
price = PRICING['price'].to_numpy()
duration = PRICING['duration'].to_numpy()
_order = PRICING['order'].to_numpy()
category = PRICING['category'].to_numpy()

#Y
quantity = PRICING['quantity'].to_numpy()
#zip sku and category

#sample/holdout creation
sku_sample = sku[np.arange(np.int(len(sku)*0.7))] 
price_sample = price[np.arange(np.int(len(price)*0.7))] 
duration_sample = duration[np.arange(np.int(len(duration)*0.7))] 
order_sample = _order[np.arange(np.int(len(_order)*0.7))] 
category_sample = category[np.arange(np.int(len(category)*0.7))] 

quantity_sample = quantity[np.arange(np.int(len(quantity)*0.7))] 

sku_holdout = sku[-np.arange(np.int(len(sku)*0.3))]
price_holdout = price[-np.arange(np.int(len(price)*0.3))]
duration_holdout = duration[-np.arange(np.int(len(duration)*0.3))]
order_holdout = _order[-np.arange(np.int(len(_order)*0.3))]
category_holdout = category[-np.arange(np.int(len(category)*0.3))]

quantity_holdout = quantity[-np.arange(np.int(len(quantity)*0.3))]

#create inputs and zipping
x_cat = np.array(list(zip(category_sample, sku_sample)))
x_num = np.array(list(zip(price_sample,order_sample,duration_sample)))

inputs_cat = tf.keras.layers.Input(shape=(2,),name= 'in_cat')
embedding = tf.keras.layers.Embedding(
    input_dim = len(np.unique(x_cat)),
    output_dim = 3,
    input_length = 6, 
    name = 'embedding')(inputs_cat)
embedding_flat = tf.keras.layers.Flatten(name='flatten')(embedding)
inputs_num = tf.keras.layers.Input(shape=(3,))
inputs_concat = tf.keras.layers.Concatenate(name = 'concatenation')([embedding_flat, inputs_num])
#elu hidden
#hidden = tf.keras.layers.Dense(70,activation = tf.keras.layers.ELU(alpha = 1.0),name = 'hidden')(inputs_concat)
#Default linear hidden
hidden = tf.keras.layers.Dense(70,name = 'hidden')(inputs_concat)
outputs = tf.keras.layers.Dense(1,name = 'out')(hidden)



ELUmodel = tf.keras.Model(inputs = [inputs_cat,inputs_num], outputs = outputs)
ELUmodel.summary()
ELUmodel.compile(loss = 'mse', optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0001, rho = 0.95, momentum = 0, epsilon = 1e-07))
#model.compile(loss = 'mse', optimizer = tf.keras.optimizers.SGD(learning_rate = 0.001))
ELUmodel.fit(x=[x_cat,x_num],y=quantity_sample, batch_size=10, epochs=5)

ELUmodel.history.history['loss']

Linmodel = tf.keras.Model(inputs = [inputs_cat,inputs_num], outputs = outputs)
Linmodel.summary()
Linmodel.compile(loss = 'mse', optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0001, rho = 0.95, momentum = 0, epsilon = 1e-07))
#model.compile(loss = 'mse', optimizer = tf.keras.optimizers.SGD(learning_rate = 0.001))
Linmodel.fit(x=[x_cat,x_num],y=quantity_sample, batch_size=10, epochs=5)

Linmodel.history.history['loss'][-1]
#RMSprop
#C:/Users/ethan/'OneDrive - University of Tennessee'/Fall_2021/BZAN544-Ballings/Group-Assignment-2/bzan554-group-assignment2

#Ethan is doing elu

#batch size: 1,5,10;epochs = 5,10;learning_rate = 0.0001,0.001,0.01;rho=0.9,0.95
batch_size = [1,5,10];epochs = [5,10];learning_rate = [0.0001,0.001,0.01];rho=[0.9,0.95]

enumeration = [(a,b,c,d) for a in batch_size for b in epochs for c in learning_rate for d in rho]
enumeration[0]


LinModelLosses = []
for x in np.arange(len(enumeration)):
    for y in np.arange(len(enumeration[0])):
        x_cat = np.array(list(zip(category_sample, sku_sample)))
        x_num = np.array(list(zip(price_sample,order_sample,duration_sample)))

        inputs_cat = tf.keras.layers.Input(shape=(2,),name= 'in_cat')
        embedding = tf.keras.layers.Embedding(
            input_dim = len(np.unique(x_cat)),
            output_dim = 3,
            input_length = 6, 
            name = 'embedding')(inputs_cat)
        embedding_flat = tf.keras.layers.Flatten(name='flatten')(embedding)
        inputs_num = tf.keras.layers.Input(shape=(3,))
        inputs_concat = tf.keras.layers.Concatenate(name = 'concatenation')([embedding_flat, inputs_num])
        #elu hidden
        #hidden = tf.keras.layers.Dense(70,activation = tf.keras.layers.ELU(alpha = 1.0),name = 'hidden')(inputs_concat)
        #Default linear hidden
        hidden = tf.keras.layers.Dense(70,name = 'hidden')(inputs_concat)
        outputs = tf.keras.layers.Dense(1,name = 'out')(hidden)
        Linmodel = tf.keras.Model(inputs = [inputs_cat,inputs_num], outputs = outputs)
        #Linmodel.summary()
        Linmodel.compile(loss = 'mse', optimizer = tf.keras.optimizers.RMSprop(learning_rate=enumeration[x][2], rho = enumeration[x][3], momentum = 0, epsilon = 1e-07))
        #model.compile(loss = 'mse', optimizer = tf.keras.optimizers.SGD(learning_rate = 0.001))
        Linmodel.fit(x=[x_cat,x_num],y=quantity_sample, batch_size=enumeration[x][1], epochs=enumeration[x][1])

        LinModelLosses.append(Linmodel.history.history['loss'][-1])