import os
import argparse
import _pickle as pickle

import speech_models
import config
import utils

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint  


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--train", action="store_true", default=False)
    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument("--data", type=str, required=False, help='Name of data folder')
    parser.add_argument("--model", type=str, required=True, help='Name of the model to save')
    parser.add_argument("--units", type=int, required=False, default=200 , help='number of the units of RNN')
    
    args = parser.parse_args()

    # make chackpoints directory, if necessary
    if not os.path.exists('../checkpoints'):
        os.makedirs('../checkpoints')

    checkpoint_path = os.path.join('../checkpoints', args.model + '.h5')
    
    ROOT = '../data'
    meta = pd.read_csv(os.path.join(ROOT,args.data ,'metadata.csv'), index_col = 'index')
    
    N_LABELS = 29

    model = speech_models.rnn_model(input_size = (config.data_detail['max_input_length'], config.data_detail['num_features']), units = args.units)

    if args.train:

        #prepare for training data
        TRAIN_STEPS = int(config.data_detail['n_training'] / config.training['batch_size'])
        VALID_STEPS = int(config.data_detail['n_valid'] / config.training['batch_size'])
        
        train_ds = utils.get_dataset_from_tfrecords(config.data_detail, tfrecords_dir=config.data_detail['data_folder'], split='train')
        valid_ds = utils.get_dataset_from_tfrecords(config.data_detail, tfrecords_dir=config.data_detail['data_folder'], split='valid')

        if os.path.isfile(checkpoint_path):
            model.load_weights(checkpoint_path)
        
        # add checkpointer
        checkpointer = ModelCheckpoint(filepath=checkpoint_path, verbose=0) 

        #train model
        history = model.fit(train_ds, epochs = config.training['epochs'], validation_data = valid_ds, validation_steps = VALID_STEPS , steps_per_epoch = TRAIN_STEPS, callbacks= [checkpointer])

        #save the result to compare models after training
        pickle_path = os.path.join('../checkpoints', args.model + '.pickle')
        with open(pickle_path, 'wb') as f:
            pickle.dump(history.history, f)

    if args.test:
        
        #prepare for testing data
        
        #load model weights
        if os.path.isfile(checkpoint):
            model.load_weights(checkpoint_path)
        
        #make predictions

        #decode predictions and save to txt file
        
        pass