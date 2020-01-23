import pandas as pd
import numpy as np
import librosa
import re

import argparse
import os
import pathlib
import shutil
from tqdm import tqdm


def clean_ipynb_folder_if_exists(folder):
    
    folder = pathlib.Path(folder)
    ipynb_paths = [str(item) for item in folder.glob('**/*') if item.is_dir() and item.name.startswith('.ipynb')]
    
    if len(ipynb_paths) > 0:
        
        for eachdir in ipynb_paths:
            shutil.rmtree(eachdir)
            print("Removed", eachdir)
    else:
        
        print('No .ipynb_checkpoints to remove')


def extract_features(file_name):
   
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        
    except Exception as e:
        print("Error encountered while parsing file: ", file_name)
        return None 
     
    return mfccs


def create_character_mapping():

    character_map = {' ': 0}

    for i in range(97, 123):
        character_map[chr(i)] = len(character_map)

    return character_map


def label_idx(label):
    
    char_2_idx = create_character_mapping() 
    
    label = ' '.join(re.split('[^a-z]', label.lower()[:-1])) #only take alpha and ignore /n at the end of a sentence
    
    vector = [char_2_idx[char] for char in label] #convert string to vector of index
    
    return vector


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, default=True)
    parser.add_argument("--dst", type=str, default=False)
    args = parser.parse_args()

    ROOT = '../data'
    SRC = os.path.join(ROOT, args.src)
    DST = os.path.join(ROOT, args.dst)
    
    if os.path.isdir(DST):
        shutil.rmtree(DST)
        os.makedirs(DST)
    else:
        os.makedirs(DST)

    clean_ipynb_folder_if_exists(SRC)
    walk_dir = list(os.walk(SRC))
    
    i = 0
    for root, dirs, metas in tqdm(walk_dir):
        
        for meta in metas:
            
            if meta[-4:] == '.txt':
                with open(os.path.join(root, meta), 'r') as f:
                    
                    for line in f.readlines():
                        
                        
                        name, label = line.split(' ', 1) 
                        
                        path = os.path.join(root, name + '.flac')                        
                        label = label_idx(label)
                        

                        if i % 100 == 0:
                            print('The number files is processed: ', i)
                        i += 1

    print('Number of processed file', i)

                        