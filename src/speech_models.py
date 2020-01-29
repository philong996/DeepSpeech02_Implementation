import os

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D, Bidirectional, LSTM, GRU, Dense, TimeDistributed
from tensorflow.keras.layers import Dropout, BatchNormalization, LeakyReLU, PReLU
from tensorflow.keras.layers import Input, MaxPooling2D, Reshape, MaxPool2D, Activation, AveragePooling2D
from tensorflow.keras.optimizers import Adam


def ctc_loss_lambda_func(y_true, y_pred):
    """Function for computing the CTC loss"""

    if len(y_true.shape) > 2:
        y_true = tf.squeeze(y_true)

    input_length = tf.math.reduce_sum(y_pred, axis=-1, keepdims=False)
    input_length = tf.math.reduce_sum(input_length, axis=-1, keepdims=True)
    label_length = tf.math.count_nonzero(y_true, axis=-1, keepdims=True, dtype="int64")

    loss = K.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    loss = tf.reduce_mean(loss)

    return loss


def rnn_model(input_size, units, layers, activation = 'relu', output_dim=29, learning_rate=3e-4):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(input_size[0], input_size[1]))
    x = input_data

    for i in range(layers):
        # Add recurrent layer
        x = GRU(units, activation=activation,
            return_sequences=True, name='rnn_{}'.format(i))(x)
        
        #Add batch normalization 
        x = BatchNormalization()(x)
    
    #Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(x)
    
    #Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    
    #Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    # model.output_length = lambda x: x
    
    #compile model
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=ctc_loss_lambda_func)
    model.summary()
    
    return model

