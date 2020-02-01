import os

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D, Conv1D, Bidirectional, LSTM, GRU, Dense, TimeDistributed
from tensorflow.keras.layers import Dropout, BatchNormalization
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


def rnn(input_size, units, layers, is_bi, activation = 'relu', output_dim=29, learning_rate=3e-4):
    """ Build a recurrent network for speech recognition
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(input_size[0], input_size[1]))

    x = BatchNormalization()(input_data)

    if is_bi:
        for i in range(layers):
            # Add recurrent layer
            x = Bidirectional(GRU(units, activation=activation,
                return_sequences=True, name='rnn_{}'.format(i+1)))(x)
            
            #Add batch normalization 
            x = BatchNormalization()(x)
    else:
        for i in range(layers):
            # Add recurrent layer
            x = GRU(units, activation=activation,
                return_sequences=True, name='rnn_{}'.format(i+1))(x)
            
            #Add batch normalization 
            x = BatchNormalization()(x)
    
    #Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(x)
    
    #Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    
    #Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    
    #compile model
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=ctc_loss_lambda_func)
    model.summary()
    
    return model


def deep_speech(input_size, units, rnn_layers, is_bi, activation = 'relu', output_dim=29, learning_rate=3e-4):
    """ Build a recurrent + convolutional network for speech 
    """
    
    # Main acoustic input
    input_data = Input(name='the_input', shape=(input_size[0], input_size[1], 1))
    x = input_data
    
    padding = (20, 5)
    x = cnn = Conv2D(filters=32, 
                     kernel_size=(41,11), 
                     strides=(2,2), 
                     padding=[[0, 0], [padding[0], padding[0]], [padding[1], padding[1]], [0, 0]])(x)
    x = BatchNormalization()(x)
    
    padding = (10, 5)
    x = cnn = Conv2D(filters=32, 
                     kernel_size=(21,11), 
                     strides=(2,1), 
                     padding=[[0, 0], [padding[0], padding[0]], [padding[1], padding[1]], [0, 0]])(x)
    x = BatchNormalization()(x)
    
    shape = x.get_shape()
    x = Reshape((shape[1], shape[2] * shape[3]))(x)
    
    # Add a recurrent layer
    if is_bi:
        for i in range(rnn_layers):
            # Add recurrent layer
            x = Bidirectional(GRU(units, activation=activation,
                return_sequences=True, name='rnn_{}'.format(i+1)))(x)
            
            #Add batch normalization 
            x = BatchNormalization()(x)
    else:
        for i in range(rnn_layers):
            # Add recurrent layer
            x = GRU(units, activation=activation,
                return_sequences=True, name='rnn_{}'.format(i+1))(x)
            
            #Add batch normalization 
            x = BatchNormalization()(x)
    
    # Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(x)
    
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    
    #compile model
    optimizer = Adam(learning_rate=learning_rate) 
    model.compile(optimizer=optimizer, loss= ctc_loss_lambda_func)
    model.summary()
    
    return model