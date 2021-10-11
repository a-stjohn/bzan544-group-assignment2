from numpy.core.fromnumeric import shape
import pandas as pd
import numpy as np
import tensorflow as tf
import os,sys

PRICING = pd.read_csv(r'C:\\Users\\ethan\\OneDrive - University of Tennessee\\Fall 2021\\BZAN544-Ballings\\pricing_final.csv')
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

#create inputs and zipping
x_cat = np.array(list(zip(category, sku)))
x_num = np.array(list(zip(price,_order,duration)))

inputs_cat = tf.keras.layers.Input(shape=(2,),name= 'in_cat')
embedding = tf.keras.layers.Embedding(
    input_dim = len(np.unique(x_cat)),
    output_dim = 3,
    input_length = 6, 
    name = 'embedding')(inputs_cat)
embedding_flat = tf.keras.layers.Flatten(name='flatten')(embedding)
inputs_num = tf.keras.layers.Input(shape=(3,))
inputs_concat = tf.keras.layers.Concatenate(name = 'concatenation')([embedding_flat, inputs_num])
hidden = tf.keras.layers.Dense(70,name = 'hidden')(inputs_concat)
outputs = tf.keras.layers.Dense(1,name = 'out')(hidden)



model = tf.keras.Model(inputs = [inputs_cat,inputs_num], outputs = outputs)
model.summary()
model.compile(loss = 'mse', optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0001, rho = 0.95, momentum = 0, epsilon = 1e-07))
#model.compile(loss = 'mse', optimizer = tf.keras.optimizers.SGD(learning_rate = 0.001))
model.fit(x=[x_cat,x_num],y=quantity, batch_size=10, epochs=5)

model.history.history['loss']

#first attempt at RMSprop; need to think about grid and function