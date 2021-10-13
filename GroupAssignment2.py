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
# create training and testing datasets

#create inputs and zipping
x_cat = np.array(list(zip(category, sku)))
x_num = np.array(list(zip(price,_order,duration)))


x_cat_train, x_cat_test = train_test_split(
    x_cat,
    test_size=0.3,
    random_state=42
)
# x_sku_train, x_sku_test = train_test_split(
#     sku,
#     test_size=0.3,
#     random_state=42
# )
# x_categ_train, x_categ_test = train_test_split(
#     category,
#     test_size=0.3,
#     random_state=42
# )


x_num_train, x_num_test = train_test_split(
    x_num,
    test_size=0.3,
    random_state=42
)
y_train, y_test = train_test_split(
    quantity,
    test_size=0.3,
    random_state=42
)

def RMSpropELU_Func(x, y, activation_input = 'sigmoid',lr_schedule=False,lr=0.001, batch_size=10, epochs=10, save_losses=True):
    """
    A function that allows the user to tune the paramaters of a deep nueral
    network with an RMSprop optimizer, and an ELU activation function.
    """
    # rmsprop model architecture
    inputs_cat = tf.keras.layers.Input(shape=(2,), name='in_cat')
    embedding = tf.keras.layers.Embedding(
        input_dim=np.max(x_cat) + 1,
        output_dim=100,
        input_length=2,
        name='embedding'
    )(inputs_cat)
    embedding_flat = tf.keras.layers.Flatten(name='flatten')(embedding)
    inputs_num = tf.keras.layers.Input(shape=(3,),name = 'in_num')
    inputs_concat = tf.keras.layers.Concatenate(
        name = 'concatenation'
    )([embedding_flat, inputs_num])
    hidden = tf.keras.layers.Dense(
        50,
        activation = activation_input,
        name = 'hidden'
    )(inputs_concat)
    outputs = tf.keras.layers.Dense(1, name = 'out')(hidden)

    model = tf.keras.Model(inputs = [inputs_cat,inputs_num], outputs = outputs)
    if lr_schedule:
        learning_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate = 1e-2,
            decay_steps = 10000,
            decay_rate = 0.9
        )
        model.compile(
            loss = 'mse',
            metrics = [tf.keras.metrics.Accuracy()],
            optimizer = tf.keras.optimizers.RMSprop(
                learning_rate=learning_schedule,
                rho=0.95,
                momentum = 0,
                epsilon = 1e-07
            )
        )
    else:
        model.compile(
            loss = 'mse',
            metrics = [tf.keras.metrics.Accuracy()],
            optimizer = tf.keras.optimizers.RMSprop(
                learning_rate=lr,
                rho=0.95,
                momentum = 0,
                epsilon=1e-07
            )
        )

    # fit the model based on func parameters
    model.fit(x=x,y=y, batch_size=batch_size, epochs=epochs)

    if save_losses:
        return model
    else:
        return model

Default_RMSprop_Sigmoid_Model = RMSpropELU_Func([x_cat_test,x_num_test],y_test)
# Default_RMSprop_Sigmoid_Model['model'].save(r'C:\Users\ethan\OneDrive - University of Tennessee\Fall_2021\BZAN544-Ballings\Group-Assignment-2\Ethan_Models\DefaultRMS_Sigmoid')
Default_RMSprop_Sigmoid_Model_Schedule = RMSpropELU_Func([x_cat_test,x_num_test],y_test,lr_schedule=True)
# Default_RMSprop_Sigmoid_Model_Schedule['model'].save(r'C:\Users\ethan\OneDrive - University of Tennessee\Fall_2021\BZAN544-Ballings\Group-Assignment-2\Ethan_Models\DfaultRMS_Sigmoid_Schedule')
RMSprop_elu_Model = RMSpropELU_Func([x_cat_test,x_num_test],y_test,activation_input='elu')
# RMSprop_elu_Model['model'].save(r'C:\Users\ethan\OneDrive - University of Tennessee\Fall_2021\BZAN544-Ballings\Group-Assignment-2\Ethan_Models\RMSprop_Elu')
RMSprop_elu_Model_Schedule = RMSpropELU_Func([x_cat_test,x_num_test],y_test,activation_input='elu',lr_schedule=True)
# RMSprop_elu_Model_Schedule['model'].save(r'C:\Users\ethan\OneDrive - University of Tennessee\Fall_2021\BZAN544-Ballings\Group-Assignment-2\Ethan_Models\RMSprop_Elu_Schedule')


DefRMSSigmoid = tf.keras.models.load_model(r'C:\Users\ethan\OneDrive - University of Tennessee\Fall_2021\BZAN544-Ballings\Group-Assignment-2\bzan554-group-assignment2\Ethan_Models\DefaultRMS_Sigmoid')
DefRMSSigmoidSched = tf.keras.models.load_model(r'C:\Users\ethan\OneDrive - University of Tennessee\Fall_2021\BZAN544-Ballings\Group-Assignment-2\bzan554-group-assignment2\Ethan_Models\DfaultRMS_Sigmoid_Schedule')
RMSProp_Elu = tf.keras.models.load_model(r'C:\Users\ethan\OneDrive - University of Tennessee\Fall_2021\BZAN544-Ballings\Group-Assignment-2\bzan554-group-assignment2\Ethan_Models\RMSprop_Elu')
RMSProp_Elu_Sched = tf.keras.models.load_model(r'C:\Users\ethan\OneDrive - University of Tennessee\Fall_2021\BZAN544-Ballings\Group-Assignment-2\bzan554-group-assignment2\Ethan_Models\RMSprop_Elu_Schedule')
import matplotlib.pyplot as plt



Default_RMSprop_Sigmoid_Model['losses']
Default_RMSprop_Sigmoid_Model_Schedule['losses']
RMSprop_elu_Model['losses']
RMSprop_elu_Model_Schedule['losses']

y_hat = DefRMSSigmoidSched.predict([x_cat_test,x_num_test])
DefRMSSigmoidSched.summary()

np.sum(np.int_(y_hat)==y_test.all()) / len(y_test)

# fig.suptitle('Model Loss over time')
#Plotting My figure
plt.plot(np.arange(len(Default_RMSprop_Sigmoid_Model['losses'])),Default_RMSprop_Sigmoid_Model['losses'],label='RMSprop & Sigmoid Activation')
plt.plot(np.arange(len(Default_RMSprop_Sigmoid_Model_Schedule['losses'])),Default_RMSprop_Sigmoid_Model_Schedule['losses'],label='RMSprop & Sigmoid & Learning Schedule')
plt.plot(np.arange(len(RMSprop_elu_Model['losses'])),RMSprop_elu_Model['losses'],label='RMSprop & elu')
plt.plot(np.arange(len(RMSprop_elu_Model_Schedule['losses'])),RMSprop_elu_Model_Schedule['losses'],label='RMSprop & elu & Learning Schedule')
plt.legend(loc='best')
plt.show()


# inputs_cat = tf.keras.layers.Input(shape=(2,),name= 'in_cat')
# #inputs_cat_sku = tf.keras.layers.Input(shape=(1,),name= 'in_sku')
# #embedding_sku = tf.keras.layers.Embedding(
# #    input_dim = np.max(x_sku_test) + 1,
# #    output_dim = 500,
# #    input_length = 1, 
# #    name = 'embedding')(inputs_cat_sku)
# #inputs_cat_categ = tf.keras.layers.Input(shape=(1,),name= 'in_categ')
# #embedding_categ = tf.keras.layers.Embedding(
# #    input_dim = np.max(x_categ_test) + 1,
# #    output_dim = 500,
# #    input_length = 1, 
# #    name = 'embedding')(inputs_cat_categ)
# embedding = tf.keras.layers.Embedding(
#     input_dim = np.max(x_cat_test) + 1,
#     output_dim = 100,
#     input_length = 2, 
#     name = 'embedding')(inputs_cat)
# embedding_flat = tf.keras.layers.Flatten(name='flatten')(embedding)
# inputs_num = tf.keras.layers.Input(shape=(3,))
# inputs_concat = tf.keras.layers.Concatenate(name = 'concatenation')([embedding_flat, inputs_num])
# #elu hidden
# #hidden = tf.keras.layers.Dense(70,activation = tf.keras.layers.ELU(alpha = 1.0),name = 'hidden')(inputs_concat)
# #Default linear hidden
# hidden = tf.keras.layers.Dense(70,name = 'hidden')(inputs_concat)
# outputs = tf.keras.layers.Dense(1,name = 'out')(hidden)


# ELUmodel = tf.keras.Model(inputs = [inputs_cat,inputs_num], outputs = outputs)
# ELUmodel.summary()
# ELUmodel.compile(loss = 'mse', optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0001, rho = 0.95, momentum = 0, epsilon = 1e-07))
# #model.compile(loss = 'mse', optimizer = tf.keras.optimizers.SGD(learning_rate = 0.001))
# ELUmodel.fit(x=[x_cat_train,x_num_train],y=y_train, batch_size=10, epochs=5)

# ELUmodel.history.history['loss']

# Linmodel = tf.keras.Model(inputs = [inputs_cat,inputs_num], outputs = outputs)
# Linmodel.summary()
# Linmodel.compile(loss = 'mse', optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0001, rho = 0.95, momentum = 0, epsilon = 1e-07))
# #model.compile(loss = 'mse', optimizer = tf.keras.optimizers.SGD(learning_rate = 0.001))
# Linmodel.fit(x=[x_cat_train,x_num_train],y=y_train, batch_size=10, epochs=5)

# Linmodel.history.history['loss'][-1]
#RMSprop
#C:/Users/ethan/'OneDrive - University of Tennessee'/Fall_2021/BZAN544-Ballings/Group-Assignment-2/bzan554-group-assignment2

#Ethan is doing elu

#batch size: 1,5,10;epochs = 5,10;learning_rate = 0.0001,0.001,0.01;rho=0.9,0.95
# batch_size = [1,5,10];epochs = [5,10];learning_rate = [0.0001,0.001,0.01];rho=[0.9,0.95]

# enumeration = [(a,b,c,d) for a in batch_size for b in epochs for c in learning_rate for d in rho]
# enumeration[0]


# LinModelLosses = []
# for x in np.arange(len(enumeration)):
#     for y in np.arange(len(enumeration[0])):
#         x_cat = np.array(list(zip(category_sample, sku_sample)))
#         x_num = np.array(list(zip(price_sample,order_sample,duration_sample)))

#         inputs_cat = tf.keras.layers.Input(shape=(2,),name= 'in_cat')
#         embedding = tf.keras.layers.Embedding(
#             input_dim = len(np.unique(x_cat)),
#             output_dim = 3,
#             input_length = 1, 
#             name = 'embedding')(inputs_cat)
#         embedding_flat = tf.keras.layers.Flatten(name='flatten')(embedding)
#         inputs_num = tf.keras.layers.Input(shape=(3,))
#         inputs_concat = tf.keras.layers.Concatenate(name = 'concatenation')([embedding_flat, inputs_num])
#         #elu hidden
#         #hidden = tf.keras.layers.Dense(70,activation = tf.keras.layers.ELU(alpha = 1.0),name = 'hidden')(inputs_concat)
#         #Default linear hidden
#         hidden = tf.keras.layers.Dense(70,name = 'hidden')(inputs_concat)
#         outputs = tf.keras.layers.Dense(1,name = 'out')(hidden)
#         Linmodel = tf.keras.Model(inputs = [inputs_cat,inputs_num], outputs = outputs)
#         #Linmodel.summary()
#         Linmodel.compile(loss = 'mse', optimizer = tf.keras.optimizers.RMSprop(learning_rate=enumeration[x][2], rho = enumeration[x][3], momentum = 0, epsilon = 1e-07))
#         #model.compile(loss = 'mse', optimizer = tf.keras.optimizers.SGD(learning_rate = 0.001))
#         Linmodel.fit(x=[x_cat,x_num],y=quantity_sample, batch_size=enumeration[x][1], epochs=enumeration[x][1])

#         LinModelLosses.append(Linmodel.history.history['loss'][-1])
