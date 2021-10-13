import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import os

os.chdir('/Users/rebeccakoch/Desktop/BZAN554/Project_2')
os.getcwd()

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

########################################################################################################

# create function
def make_a_model(x, y, optimizer, activation_func='sigmoid',
                  lr_schedule=False, lr=0.001, batch_size=10,
                  epochs=10, save_losses=True):
    """
    A function that allows the user to tune the paramaters of a deep nueral
    network with an AdaGrad optimizer.
    """
    # adagrad model architecture
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
        if optimizer == 'Adagrad':
            model.compile(
                loss = 'mse',
                optimizer = tf.keras.optimizers.Adagrad(
                    learning_rate=learning_schedule,
                    initial_accumulator_value=0.1,
                    epsilon=1e-07
                )
            )
        elif optimizer == 'Adam':
            model.compile(
                loss = 'mse',
                optimizer = tf.keras.optimizers.Adam(
                    learning_rate=learning_schedule,
                    beta_1=0.9,
                    beta_2=0.999,
                    epsilon=1e-07
                )
            )
        elif optimizer == 'RMSprop':
            model.compile(
                loss = 'mse',
                optimizer = tf.keras.optimizers.RMSprop(
                    learning_rate=learning_schedule,
                    rho=0.95,
                    momentum = 0,
                    epsilon=1e-07
                )
            )
        elif optimizer == 'Nadam':
            model.compile(
                    loss = 'mse',
                    optimizer = tf.keras.optimizers.Nadam(
                        learning_rate=learning_schedule,
                        beta_1= 0.9,
                        beta_2= 0.999,
                        epsilon=1e-07,
                        name = 'Nadam'
                    )
            )
    else:
        if optimizer == 'Adagrad':
            model.compile(
                loss = 'mse',
                optimizer = tf.keras.optimizers.Adagrad(
                    learning_rate=lr,
                    initial_accumulator_value=0.1,
                    epsilon=1e-07
                )
            )
        elif optimizer == 'Adam':
            model.compile(
                loss = 'mse',
                optimizer = tf.keras.optimizers.Adam(
                    learning_rate=lr,
                    beta_1=0.9,
                    beta_2=0.999,
                    epsilon=1e-07
                )
            )
        elif optimizer == 'RMSprop':
            model.compile(
                loss = 'mse',
                optimizer = tf.keras.optimizers.RMSprop(
                    learning_rate=lr,
                    rho=0.95,
                    momentum = 0,
                    epsilon=1e-07
                )
            )
        elif optimizer == 'Nadam':
            model.compile(
                    loss = 'mse',
                    optimizer = tf.keras.optimizers.Nadam(
                        learning_rate=lr,
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

adam_schedule_elu_model = make_a_model(
    x = [x_cat_train, x_num_train],
    y = y_train,
    activation_func = 'elu',
    lr_schedule = True,
    optimizer = 'Adam',
    batch_size = 50
)
adam_schedule_elu_model['model'].save('./becky_models/adam_schedule_elu_model')

########################################################################################################

# RUNNING FUNCTION W/ DIFF PARAMS
# adam optimizer, sigmoid activation, no learning schedule
#adam_base_model = adam_model(x = [x_cat_train, x_num_train], y = y_train) 
# loss: 788.0125

# adam optimizer, sigmoid activation, learning schedule
#adam_schedule_model = adam_model(x = [x_cat_train, x_num_train], y = y_train, lr_schedule=True)
# loss: 613.9511

# adam optimizer, ReLU activation, no learning schedule
#adam_relu_model = adam_model(x = [x_cat_train, x_num_train], y = y_train, activation_func='relu')
# loss: 775.7158

# adam optimizer, ReLU activation, learning schedule
#adam_schedule_relu_model = adam_model(x = [x_cat_train, x_num_train], y = y_train, activation_func='relu', lr_schedule=True)
# loss: 596.8219

# adam optimizer, ELU activation, no learning schedule
#adam_elu_model = adam_model(x = [x_cat_train, x_num_train], y = y_train, activation_func='elu')
# loss: 777.1718

# adam optimizer, ELU activation, learning schedule
#adam_schedule_elu_model = adam_model(x = [x_cat_train, x_num_train], y = y_train, activation_func='elu', lr_schedule=True)
# loss: 


