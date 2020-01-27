import pandas as pd
import numpy as np
import librosa
import re

import datetime
import codecs
import csv
import os
import pathlib
import shutil
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

import config

def clean_ipynb_folder_if_exists(folder):
    
    folder = pathlib.Path(folder)
    ipynb_paths = [str(item) for item in folder.glob('**/*') if item.is_dir() and item.name.startswith('.ipynb')]
    
    if len(ipynb_paths) > 0:
        
        for eachdir in ipynb_paths:
            shutil.rmtree(eachdir)
            print("Removed", eachdir)
    else:
        
        print('No .ipynb_checkpoints to remove')


def create_token_index():
    vocab_file = './vocab.txt'
    lines = []
    
    #read vocab txt
    with codecs.open(vocab_file, "r", "utf-8") as fin:
        lines.extend(fin.readlines())
    
    token_to_index = {}
    index_to_token = {}
    index = 0
    
    for line in lines:
        line = line[:-1]  # Strip the '\n' char.
        if line.startswith("#"):
            # Skip from reading comment line.
            continue
        token_to_index[line] = index
        index_to_token[index] = line
        index += 1
    return token_to_index, index_to_token

def label_idx(label):
    
    token_to_index, _ = create_token_index() 
    
    tokens = list(label.lower().strip())

    #convert string to vector of index
    labels = [token_to_index[char] for char in tokens] 
    
    return labels

def idx_string(indices):
    _, index_to_token = create_token_index()
    
    #create a string from indices
    string = ''.join([index_to_token[index] for index in indices]) 
    
    return string

def create_main_metadata(SRC, DST):
    clean_ipynb_folder_if_exists(SRC)
    walk_dir = list(os.walk(SRC))

    i = 0
    
    if os.path.isdir(DST):
        shutil.rmtree(DST)
        os.makedirs(DST)
    else:
        os.makedirs(DST)
    
    with open(os.path.join(DST, 'metadata.csv'), 'w', newline='') as metadata:
        
        metadata_writer = csv.DictWriter(metadata, delimiter=',', fieldnames=['index','filepath', 'label', 'label_length', 'spec_length'])
        metadata_writer.writeheader()
    
        for root, dirs, metas in tqdm(walk_dir):

            for meta in metas:

                if meta[-4:] == '.txt':
                    with open(os.path.join(root, meta), 'r') as f:

                        for line in f.readlines():

                            name, label = line.split(' ', 1) 
                            path = os.path.join(root, name + '.flac')
                            
                            spec_length = extract_features(path).shape[1]
                            label = label_idx(label)
                            
                            metadata_writer.writerow({
                                'index':i,
                                'filepath': path,
                                'label': label,
                                'label_length': len(label),
                                'spec_length': spec_length
                            })

                            i += 1

        print('Number of processed file', i)


def _float_feature(list_of_floats):  # float32
    return tf.train.Feature(float_list=tf.train.FloatList(value=list_of_floats))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def extract_features(file_path):
    try:
        audio, sample_rate = librosa.load(file_path) 
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        
    except Exception as e:
        print("Error encountered while parsing file: ", file_path)
        return None 
    
    return mfccs


class TFRecordsConverter:
    """Convert audio to TFRecords."""
    def __init__(self, meta_path, output_dir, test_size, val_size):
        self.output_dir = output_dir

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        df = pd.read_csv(meta_path, index_col=0)
        # Shuffle data by "sampling" the entire data-frame
        self.df = df.sample(frac=1, random_state=101)

        self.max_input_len = self.df.spec_length.max()
        self.max_label_len = self.df.label_length.max()
        
        n_samples = len(df)
        self.n_test = int(np.ceil(n_samples * test_size))
        self.n_val = int(np.ceil(n_samples * val_size))
        self.n_train = int(n_samples - self.n_test - self.n_val)
        
        self.n_shards_train = int(max(1000, self.n_train) / 1000)
        self.n_shards_test = int(max(1000, self.n_test) / 1000)
        self.n_shards_val = int(max(1000, self.n_val) / 1000)
        
    def _get_shard_path(self, split, shard_id, shard_size):
        return os.path.join(self.output_dir,
                            '{}-{:03d}-{}.tfrecord'.format(split, shard_id,
                                                           shard_size))

    def _write_tfrecord_file(self, shard_path, indices):
        """Write TFRecord file."""
        with tf.io.TFRecordWriter(shard_path, options='ZLIB') as out:
            
            for index in indices:
                
                index = int(index)
                file_path = self.df.filepath.iloc[index]
                
                label = eval(self.df.label.iloc[index])
                
                if len(label) < self.max_label_len:
                    offset = self.max_label_len - len(label)
                    padding = [0 for _ in range(offset)]
                    label = label + padding
                
                feature = extract_features(file_path)
                feature = pad_sequences(feature, maxlen= self.max_input_len, padding = 'post')
                feature = feature.T
                # Example contains two features: A FloatList for the decoded
                # audio data and convert them to MFCC and an Int64List containing the corresponding
                # label's index.
                example = tf.train.Example(features=tf.train.Features(feature={
                    'feature': _float_feature(feature.flatten().tolist()),
                    'label': _int64_feature(label)}))

                out.write(example.SerializeToString())

    def convert(self):
        """Convert to TFRecords.
        Partition data into training, testing and validation sets. Then,
        divide each data set into the specified number of TFRecords shards.
        """
        splits = ('train', 'test', 'valid')
        split_sizes = (self.n_train, self.n_test, self.n_val)
        split_n_shards = (self.n_shards_train, self.n_shards_test,
                          self.n_shards_val)
        
        start_time = datetime.datetime.now()
        offset = 0
        for split, size, n_shards in zip(splits, split_sizes, split_n_shards):
            
            print('Converting {} set into TFRecord shards...'.format(split))
            shard_size = np.ceil(size / n_shards)
            cumulative_size = offset + size
            
            for shard_id in tqdm(range(1, n_shards + 1)):
                step_size = min(shard_size, cumulative_size - offset)
                shard_path = self._get_shard_path(split, shard_id, step_size)
                
                # Generate a subset of indices to select only a subset of
                # audio-files/labels for the current shard.
                file_indices = np.arange(offset, offset + step_size)
                self._write_tfrecord_file(shard_path, file_indices)
                offset += step_size
        
        total_time = datetime.datetime.now() - start_time
        
        data_info = "\n".join([
            'Number of training examples:       {}'.format(self.n_train),
            'Number of testing examples:        {}'.format(self.n_test),
            'Number of validation examples:     {}'.format(self.n_val),
            'TFRecord files saved to:           {}'.format(self.output_dir),
            'Number of examples:                {}'.format(self.df.shape[0]),
            'Max input length:                  {}'.format(self.max_input_len),
            'Max label length:                  {}'.format(self.max_label_len),
            'Total time for processing:         {}'.format(total_time),
            'Number of shards for training:     {}'.format(self.n_shards_train),
            'Number of shards for testing:      {}'.format(self.n_shards_test),
            'Number of shards for valid:        {}'.format(self.n_shards_valid)
        ])

        ROOT = os.path.dirname(self.output_dir)
        
        with open(os.path.join(ROOT,"data_info.txt"), "w") as f:
            f.write(data_info)
            print(data_info)


#Each tf.train.Example record contains one or more "features", and the input pipeline typically converts these features into tensors.
def _parse_batch(record_batch, training_config):

    # Create a description of the features
    feature_description = {
        'feature': tf.io.FixedLenFeature([training_config['max_input_length'],training_config['num_features']], tf.float32),
        'label': tf.io.FixedLenFeature([training_config['max_label_length']], tf.int64)
    }

    # Parse the input `tf.Example` proto using the dictionary above
    example = tf.io.parse_example(record_batch, feature_description)
    
    return example['feature'], example['label']


def get_dataset_from_tfrecords(training_config, tfrecords_dir='tfrecords' , split='train', batch_size=64):
    
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    
    if split not in ('train', 'test', 'valid'):
        raise ValueError("split must be either 'train', 'test' or 'valid'")
    
    # List all *.tfrecord files for the selected split
    pattern = os.path.join(tfrecords_dir, '{}*.tfrecord'.format(split))
    files_ds = tf.data.Dataset.list_files(pattern)

    # Disregard data order in favor of reading speed
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False
    files_ds = files_ds.with_options(ignore_order)

    # Read TFRecord files in an interleaved order
    ds = tf.data.TFRecordDataset(files_ds,
                                 compression_type='ZLIB')
    
    # Prepare batches
    ds = ds.batch(batch_size)

    # Parse a batch into a dataset of [audio, label] pairs
    ds = ds.map(lambda x: _parse_batch(x, training_config))

    ds = ds.repeat()

    return ds.prefetch(buffer_size=AUTOTUNE)