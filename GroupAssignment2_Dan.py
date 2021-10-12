import pandas as pd
import numpy as np
import tensorflow as tf

# inspect some the data
data = pd.read_csv('pricing_final.csv')
data.head()

# explanatory variables
sku = data['sku'].to_numpy()
price = data['price'].to_numpy()
_order = data['order'].to_numpy()
duration = data['duration'].to_numpy()
category = data['category'].to_numpy()

# response variable
quantity = data['quantity'].to_numpy()

# Create out categorical and numeric inputs (zipping stuff together)
x_cat = np.array(list(zip(category, sku)))
x_num = np.array(list(zip(price, _order, duration)))

inputs_cat = tf.keras.layers.Input(shape=(2,), name='in_cat')
embedding = tf.keras.layers.Embedding(
    input_dim=len(np.unique(x_cat)),
    output_dim=5,
    input_length=2,
    name='embedding'
)(inputs_cat)
embedding_flat = tf.keras.layers.Flatten(name='flatten')(embedding)
inputs_num = tf.keras.layers.Input(shape=(3,),name = 'in_num')
inputs_concat = tf.keras.layers.Concatenate(
    name = 'concatenation'
)([embedding_flat, inputs_num])
hidden = tf.keras.layers.Dense(
    200,
    name='hidden1',
    activation = "tanh"
    )(inputs_concat)
outputs = tf.keras.layers.Dense(1, name = 'out')(hidden)

tanh_Nadam_model = tf.keras.Model(inputs = [inputs_cat,inputs_num], outputs = outputs)
tanh_Nadam_model.summary()

# loss: 1604.5897
initial_learning_rate = 0.1
decay_steps = 10000
decay_rate = 1/10
learning_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps, decay_rate
)
tanh_Nadam_model.compile(
    loss = 'mse',
    optimizer = tf.keras.optimizers.Nadam(
        learning_rate=0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-07, name = "Nadam")
)
tanh_Nadam_model.fit(x=[x_cat,x_num],y=quantity, batch_size=10, epochs=10)

tanh_Nadam_model.history.history['loss']

tf.keras.models.save_model(
    tanh_Nadam_model,
    filepath = 'C:\\Users\\danie\\Documents\\GitHub\\bzan554-group-assignment2\\models'
)