import pandas as pd
import numpy as np
import librosa
import re
import soundfile
import editdistance
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
                            
                            au, sr = soundfile.read(path)
                            spec_length = compute_spectrogram_feature(au, sr).shape[0]
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


def compute_spectrogram_feature(samples, sample_rate, stride_ms=10.0,
                                window_ms=20.0, max_freq=None, eps=1e-14):
    """Compute the spectrograms for the input samples(waveforms).
    the code is from tensorflow research
    """
    if max_freq is None:
        max_freq = sample_rate / 2
    if max_freq > sample_rate / 2:
        raise ValueError("max_freq must not be greater than half of sample rate.")

    if stride_ms > window_ms:
        raise ValueError("Stride size must not be greater than window size.")

    stride_size = int(0.001 * sample_rate * stride_ms)
    window_size = int(0.001 * sample_rate * window_ms)

    # Extract strided windows
    truncate_size = (len(samples) - window_size) % stride_size
    samples = samples[:len(samples) - truncate_size]
    nshape = (window_size, (len(samples) - window_size) // stride_size + 1)
    nstrides = (samples.strides[0], samples.strides[0] * stride_size)
    windows = np.lib.stride_tricks.as_strided(
      samples, shape=nshape, strides=nstrides)
    assert np.all(
      windows[:, 1] == samples[stride_size:(stride_size + window_size)])

    # Window weighting, squared Fast Fourier Transform (fft), scaling
    weighting = np.hanning(window_size)[:, None]
    fft = np.fft.rfft(windows * weighting, axis=0)
    fft = np.absolute(fft)
    fft = fft**2
    scale = np.sum(weighting**2) * sample_rate
    fft[1:-1, :] *= (2.0 / scale)
    fft[(0, -1), :] /= scale
    # Prepare fft frequency list
    freqs = float(sample_rate) / window_size * np.arange(fft.shape[0])

    # Compute spectrogram feature
    ind = np.where(freqs <= max_freq)[0][-1] + 1
    specgram = np.log(fft[:ind, :] + eps)
    return np.transpose(specgram, (1, 0))


def normalize_audio_feature(audio_feature):
  """Perform mean and variance normalization on the spectrogram feature.
  Args:
    audio_feature: a numpy array for the spectrogram feature.
  Returns:
    a numpy array of the normalized spectrogram.
  """
  mean = np.mean(audio_feature, axis=0)
  var = np.var(audio_feature, axis=0)
  normalized = (audio_feature - mean) / (np.sqrt(var) + 1e-6)

  return normalized


def mfcc_feature(file_path):
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
        
        self.n_shards_train = int(np.ceil(max(1000, self.n_train) / 1000))
        self.n_shards_test = int(np.ceil(max(1000, self.n_test) / 1000))
        self.n_shards_val = int(np.ceil(max(1000, self.n_val) / 1000))
        
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
                
                au, sr = soundfile.read(file_path)
                feature = compute_spectrogram_feature(au, sr)
                feature = normalize_audio_feature(feature)
                feature = pad_sequences(feature.T, maxlen= self.max_input_len, padding = 'post')
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
        
        with open(os.path.join(os.path.dirname(self.output_dir), 'labels.csv'), 'w', newline='') as f:
            labels_file = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            
            labels_file.writerow(['labels', 'split'])
            
            for i in range(len(self.df)):
                if i < self.n_train:
                    labels_file.writerow([self.df.label.iloc[i], 'train'])
                elif i < self.n_train + self.n_test:
                    labels_file.writerow([self.df.label.iloc[i], 'test'])
                else:
                    labels_file.writerow([self.df.label.iloc[i], 'valid'])

        start_time = datetime.datetime.now()
        
        offset = 0
        for split, size, n_shards in zip(splits, split_sizes, split_n_shards):
            
            if size == 0:
                continue
            
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
            'Number of shards for valid:        {}'.format(self.n_shards_val)
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


def get_dataset_from_tfrecords(training_config, tfrecords_dir , split='train', batch_size=32):
    
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    
    #take labels
    ROOT = os.path.dirname(tfrecords_dir)
    label_file = os.path.join(ROOT, 'labels.csv')
    df_labels = pd.read_csv(label_file)

    labels = df_labels['labels'][df_labels['split'] == split].to_list()
    labels = [idx_string(eval(label)) for label in labels]


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

    return ds.prefetch(buffer_size=AUTOTUNE), labels

def calculate_metrics(predicts, ground_truth):
    """Calculate Character Error Rate (CER), Word Error Rate (WER) and Sequence Error Rate (SER)"""

    if len(predicts) == 0 or len(ground_truth) == 0:
        return (1, 1, 1)

    cer, wer, ser = [], [], []

    for (pd, gt) in zip(predicts, ground_truth):
        pd_cer, gt_cer = list(pd.lower()), list(gt.lower())
        dist = editdistance.eval(pd_cer, gt_cer)
        cer.append(dist / (max(len(pd_cer), len(gt_cer))))

        pd_wer, gt_wer = pd.lower().split(), gt.lower().split()
        dist = editdistance.eval(pd_wer, gt_wer)
        wer.append(dist / (max(len(pd_wer), len(gt_wer))))

        pd_ser, gt_ser = [pd], [gt]
        dist = editdistance.eval(pd_ser, gt_ser)
        ser.append(dist / (max(len(pd_ser), len(gt_ser))))

    cer_f = sum(cer) / len(cer)
    wer_f = sum(wer) / len(wer)
    ser_f = sum(ser) / len(ser)

    return (cer_f, wer_f, ser_f)

def plot_stats(training_stats, val_stats, x_label='Training Steps', stats='loss'):
    stats, x_label = stats.title(), x_label.title()
    legend_loc = 'upper right' if stats=='loss' else 'lower right'
    training_steps = len(training_stats)
    test_steps = len(val_stats)

    plt.figure()
    plt.ylabel(stats)
    plt.xlabel(x_label)
    plt.plot(training_stats, label='Training' + stats)
    plt.plot(np.linspace(0, training_steps, test_steps), val_stats, label='Validation' + stats)
    plt.ylim([0,max(plt.ylim())])
    plt.legend(loc=legend_loc)
    plt.show()