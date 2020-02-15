import os
import re
import shutil
import pathlib
from tqdm import tqdm
import argparse

from pydub import AudioSegment
import pandas as pd
import pickle


def trim_audio(audio, sentence):
    start = sentence['start'] * 1000
    duration = sentence['duration'] * 1000
    stop = start + duration
    
    return audio[start:stop]

    
def preprocess_sent(audio_idx, parent_dir, folder_path = None):
    
    #read the transcript pickle file
    transcript_path = os.path.join(parent_dir, str(audio_idx), 'transcript.pickle')   
    with open(transcript_path, 'rb') as f:
        transcript = pickle.load(f)
    
    #convert transcript to dataframe
    trans_df = pd.DataFrame(transcript)
    
    #replace (applause) or (laughter) from transcript
    trans_df.text = trans_df.text.str.replace(r'\(\w+\)', '').str.strip()
    #remove na
    trans_df = trans_df.dropna()
    #get only sentences with more than 1 words
    trans_df = trans_df[trans_df.text.str.split().apply(len) > 1]
    #add links for sentence
    trans_df['sent_path'] = trans_df.apply(lambda x:  os.path.join(folder_path,'sentence_{}.wav'.format(x.name)) , axis = 1)
    
    #save the final version
    final_trans_path = os.path.join(parent_dir, str(audio_idx), 'final_transcript.csv')
    trans_df.to_csv(final_trans_path)
    
    return trans_df
    

def split_audio_trans(audio_idx, parent_dir):
    
    #preprocess the folder
    folder_path = os.path.join(parent_dir, str(audio_idx), 'sent_audio')

    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.mkdir(folder_path)
 
    #read the audio
    audio_path = os.path.join(parent_dir, str(audio_idx) , '{}.wav'.format(audio_idx))
    audio = AudioSegment.from_wav(audio_path)
    
    #get the transcript dataframe
    trans_df = preprocess_sent(audio_idx, parent_dir, folder_path)
    
    #split the sentences from the transcript
    for sent_id, sentence in trans_df.iterrows():
        #pass the sentence with just 1 word
        if len(sentence['text'].split()) < 2:
            continue
            
        #get the duration to slice the audio
        sent_audio = trim_audio(audio, sentence)
        
        #export the audio of sentence 
        sent_path = os.path.join(folder_path,'sentence_{}.wav'.format(sent_id))
        sent_audio.export(sent_path, format='wav')


def clean_ipynb_folder_if_exists(folder):
    
    folder = pathlib.Path(folder)
    ipynb_paths = [str(item) for item in folder.glob('**/*') if item.is_dir() and item.name.startswith('.ipynb')]
    
    if len(ipynb_paths) > 0:
        
        for eachdir in ipynb_paths:
            shutil.rmtree(eachdir)
            print("Removed", eachdir)
    else:
        
        print('No .ipynb_checkpoints to remove')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--audio', type=str, required=False, default='talks_for_app')
    
    args = parser.parse_args()

    PARENT_DIR = os.path.join('..','data',args.audio)
    
    clean_ipynb_folder_if_exists(PARENT_DIR)
    
    audio_indices = list(os.walk(PARENT_DIR))[0][1]

    for audio_idx in tqdm(audio_indices):
        split_audio_trans(audio_idx, PARENT_DIR)
