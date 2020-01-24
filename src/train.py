import os
import argparse

import speech_models
import config
import utils

import tensorflow as tf
import pandas as pd
import numpy as np



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--train", action="store_true", default=False)
    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument("--name", type=str, required=False, help='Name of data folder')

    args = parser.parse_args()


    checkpoint = '../checkpoints/checkpoint_weights.hdf5'   
    
    ROOT = '../data'
    meta = pd.read_csv(os.path.join(ROOT,args.name ,'metadata.csv'), index_col = 'index')

    data_detail = utils.get_data_detail(meta)

    N_ROWS = 40
    N_COLS = data_detail['max_input_length']
    N_SAMPLES = data_detail['num_samples']
    N_LABELS = 28
    BATCH_SIZE = config.training['batch_size']
    EPOCHS = config.training['epochs']
    STEPS = int(N_SAMPLES / BATCH_SIZE)

    model = speech_models.build_baseline_model(input_size = (N_COLS, N_ROWS, 1), d_model = N_LABELS)

    if args.train:

        #prepare for training data
        train_ds = utils.get_dataset_from_tfrecords(data_detail, tfrecords_dir=os.path.join(ROOT, args.name, 'TFrecords'), split='train')

        model.fit(train_ds, epochs = EPOCHS, steps_per_epoch = STEPS)