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

def adagrad_model(x, y, activation_func='sigmoid', lr_schedule=False,
                  lr=0.001, batch_size=10, epochs=10, save_losses=True):
    """
    A function that allows the user to tune the paramaters of a deep nueral
    network with an AdaGrad optimizer.
    """
    # adagrad model architecture
    inputs_cat = tf.keras.layers.Input(shape=(2,), name='in_cat')
    embedding = tf.keras.layers.Embedding(
        input_dim=len(np.unique(x_cat)),
        output_dim=6,
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
        activation = activation_func,
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
            optimizer = tf.keras.optimizers.Adagrad(
                learning_rate=learning_schedule,
                initial_accumulator_value=0.1,
                epsilon=1e-07
            )
        )
    else:
        model.compile(
            loss = 'mse',
            optimizer = tf.keras.optimizers.Adagrad(
                learning_rate=lr,
                initial_accumulator_value=0.1,
                epsilon=1e-07
            )
        )

    # fit the model based on func parameters
    model.fit(x=x,y=y, batch_size=batch_size, epochs=epochs)

    if save_losses:
        return {'model': model, 'losses': model.history.history['loss']}
    else:
        return model

# model with sigmoid activation, lr default 0.001, batch size 10, and epochs 10
base_model = adagrad_model(x = [x_cat_train, x_num_train], y = y_train)
# save the model so we don't have to train everytime
base_model['model'].save('./aaron_models/base_model')

# model with sigmoid activation, lr exp decay schedule, batch size 10, and
# epochs 10
base_schedule_model = adagrad_model(
    x = [x_cat_train, x_num_train],
    y = y_train,
    lr_schedule = True
)
base_schedule_model['model'].save('./aaron_models/base_schedule_model')

# model with relu activation, lr default 0.001, batch size 10, and epochs 10
relu_model = adagrad_model(
    x = [x_cat_train, x_num_train],
    y = y_train,
    activation_func = 'relu'
)
relu_model['model'].save('./aaron_models/relu_model')

# model with relu activation, lr exp decay schedule, batch size 10, and
# epochs 10
relu_schedule_model = adagrad_model(
    x = [x_cat_train, x_num_train],
    y = y_train,
    lr_schedule = True,
    activation_func = 'relu'
)
relu_schedule_model['model'].save('./aaron_models/relu_schedule_model')