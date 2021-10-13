import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# inspect some the data
data = pd.read_csv('pricing_final.csv')
data.head()

# explanatory variables
sku = data['sku'].to_numpy()
price = data['price'].to_numpy()
_order = data['order'].to_numpy()
duration = data['duration'].to_numpy()
category = data['category'].to_numpy()
len(np.unique(category))
len(np.unique(sku))
# response variable
quantity = data['quantity'].to_numpy()

# Create out categorical and numeric inputs (zipping stuff together)
x_cat = np.array(list(zip(category, sku)))
x_num = np.array(list(zip(price, _order, duration)))

# create training and testing datasets
x_cat_train, x_cat_test = train_test_split(
    x_cat,
    test_size=0.3,
    random_state=42
)
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


def nadam_model(x, y, activation_func='sigmoid', batch_size=10,
                epochs=10, save_losses=True, simple: bool = True,
                complex: bool = False):
    """
    A function that allows the user to tune the paramaters of a deep nueral
    network with an Nadam optimizer.
    """
    # Adam model architecture
    if simple:
        print('Using Simple Model')
        inputs_cat = tf.keras.layers.Input(shape=(2,), name='in_cat')
        embedding = tf.keras.layers.Embedding(
            input_dim=len(np.unique(x_cat)),
            output_dim=5,
            input_length=2,
            name='embedding'
        )(inputs_cat)
        embedding_flat = tf.keras.layers.Flatten(name='flatten')(embedding)
        inputs_num = tf.keras.layers.Input(shape=(3,), name='in_num')
        inputs_concat = tf.keras.layers.Concatenate(
            name='concatenation'
        )([embedding_flat, inputs_num])
        hidden = tf.keras.layers.Dense(
            200,
            activation = activation_func,
            name = 'hidden'
        )(inputs_concat)
        outputs = tf.keras.layers.Dense(1, name = 'out')(hidden)

        model = tf.keras.Model(inputs = [inputs_cat,inputs_num], outputs = outputs)
    if complex:
        print('Using Complex Model')
        '''
        This is a more complex model with more 
        '''
        inputs_cat = tf.keras.layers.Input(shape=(2,), name='in_cat')
        embedding = tf.keras.layers.Embedding(
            input_dim=len(np.unique(x_cat)),
            output_dim=200,
            input_length=2,
            name='embedding'
        )(inputs_cat)
        embedding_flat = tf.keras.layers.Flatten(name='flatten')(embedding)
        inputs_num = tf.keras.layers.Input(shape=(3,),name = 'in_num')
        inputs_concat = tf.keras.layers.Concatenate(
            name = 'concatenation'
        )([embedding_flat, inputs_num])
        hidden1 = tf.keras.layers.Dense(
            100,
            activation = activation_func,
            name = 'hidden1'
        )(inputs_concat)
        hidden2 = tf.keras.layers.Dense(
            50,
            activation = activation_func,
            name = 'hidden2'
        )(hidden1)
        outputs = tf.keras.layers.Dense(1, name = 'out')(hidden2)

        model = tf.keras.Model(inputs = [inputs_cat,inputs_num], outputs = outputs)
        batch_size = int(batch_size/2)
    
    model.compile(
        loss = 'mse',
        optimizer = tf.keras.optimizers.Nadam(
            learning_rate=0.001,
            beta_1= 0.9,
            beta_2= 0.999,
            epsilon=1e-07,
            name = 'Nadam'
        )
    )


    # fit the model based on func parameters
    model.fit(x=x,y=y, batch_size=batch_size, epochs=epochs)

    if save_losses:
        return {'model': model, 'losses': model.history.history['loss']}
    else:
        return model

# model with sigmoid activation, lr default 0.001, batch size 10, and epochs 10
complex_base_model = nadam_model(
        x = [x_cat_train, x_num_train], 
        y = y_train, 
        activation_func = 'relu', 
        simple=True)

# Save this bro
complex_base_model['model'].save('./dan_models/nadam_relu_model')