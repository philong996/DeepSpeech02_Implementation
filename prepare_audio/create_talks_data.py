import os
import shutil
import argparse

import pandas as pd
import pickle

from youtube_search import YoutubeSearch
from youtube_transcript_api import YouTubeTranscriptApi
import pafy 


def prepare_talks_data(num_talks, ted_main_dir):
    ted_path = os.path.join(ted_main_dir,'ted_main.csv')

    ##read ted_main from data of kaggle
    ted_main = pd.read_csv(ted_path)

    #take talks of only 1 speaker
    talks_data = ted_main[ted_main.num_speaker == 1]

    #sort videos according to the number of views
    talks_data = talks_data.sort_values(by='views', ascending = False)

    #just get neccesary information
    talks_data = talks_data[['name','title','main_speaker','description','duration','ratings','tags']].head(num_talks)

    talks_data.reset_index()
    
    return talks_data


def search_id(name):
    try:
        search_result = YoutubeSearch(name, max_results=1).to_json()
        video_id = eval(search_result)['videos'][0]['id']
        return video_id
    except:
        return None


def get_transcript(youtube_idx, idx ,parent_dir):
    talks_path = os.path.join(parent_dir, str(idx))

    if os.path.exists(talks_path):
        shutil.rmtree(talks_path)

    #create folder to store the audio and transcript
    os.mkdir(talks_path)
    
    #get transcript by youtube api
    transcript = YouTubeTranscriptApi.get_transcript(youtube_idx)

    #save the transcript to pickle file
    trans_path = os.path.join(talks_path,'transcript.pickle')
    with open(trans_path, 'wb') as f:
        pickle.dump(transcript, f)

def get_audio(youtube_idx, idx ,parent_dir):
    url = 'https://www.youtube.com/watch?v={}'.format(youtube_idx)
    video = pafy.new(url) 

    audio_path = os.path.join(parent_dir, str(idx), '{}.webm'.format(idx))
 
    bestaudio = video.getbestaudio() 
    bestaudio.download(filepath = audio_path) 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--num-talks', type=int, required=True)
    parser.add_argument('--ted-main', default = os.path.join('..','data','ted_talks_dataset') ,type=str, required=False)    

    args = parser.parse_args()
    
    talks_data = prepare_talks_data(args.num_talks, args.ted_main)
    
    PARENT_DIR = os.path.join('..','data','talks_for_app')

    #get names of audios
    names = talks_data.name.to_list()

    youtube_indices = []

    for idx, name in enumerate(names):
        #get video index
        youtube_idx = search_id(name)

        #add youtube index to indices
        youtube_indices.append(youtube_idx)
        
        #save transcript
        get_transcript(youtube_idx, idx ,PARENT_DIR)

        #down audio from talk
        get_audio(youtube_idx, idx ,PARENT_DIR)

        #convert the webm file to wav file
        os.system("ffmpeg -i {0}.webm -ac 1 -ar 16000 {0}.wav".format(os.path.join(PARENT_DIR, str(idx), str(idx))))
        os.system("rm {}.webm".format(os.path.join(PARENT_DIR, str(idx), str(idx))))

    #save the final result
    talks_data['youtube_idx'] = youtube_indices
    talks_data.to_csv(os.path.join(PARENT_DIR,'talks_data.csv'),index=False)
    