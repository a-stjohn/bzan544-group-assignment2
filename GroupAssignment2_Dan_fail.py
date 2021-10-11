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

x_cat_category = np.array(list(category))
x_cat_sku = np.array(list(sku))

# Creating Category Inputs
input_category = tf.keras.layers.Input(
    shape = (1,),
    name = 'in_category')
embedding_category = tf.keras.layers.Embedding(
    input_dim = len(np.unique(x_cat_category)),
    output_dim = 32,
    input_length = 1,
    name = 'embed_cat')(input_category)
embed_flat_cat = tf.keras.layers.Flatten(name = 'flat_cat')(embedding_category)

# Creating SKU Inputs
input_sku = tf.keras.layers.Input(
    shape = (1,),
    name = 'in_sku')
embedding_sku = tf.keras.layers.Embedding(
    input_dim = len(np.unique(x_cat_sku)),
    output_dim = 5000,
    input_length = 1,
    name = 'embed_sku')(input_sku)
embed_flat_sku = tf.keras.layers.Flatten(name = 'flat_sku')(embedding_sku)

inputs_num = tf.keras.layers.Input(shape = (3,), name = 'in_num')

inputs_concat = tf.keras.layers.Concatenate(name = 'concatenation')(
    [embed_flat_cat,embed_flat_sku, inputs_num])

hidden1 = tf.keras.layers.Dense(
    units = 7000,
    name = 'hidden1'
)(inputs_concat)

outputs = tf.keras.layers.Dense(1, name = 'out')(hidden1)

model = tf.keras.Model(inputs = [inputs_num, inputs_concat], outputs = outputs)
model.summary()
model.compile(loss = 'mse', optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001))
model.fit(x = [x_cat, x_num], y = quantity, batch_size = 1, epochs = 5)










