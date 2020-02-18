from flask import Blueprint, render_template, Response
from flask_paginate import Pagination, get_page_args
import csv
import json
from flask import jsonify, make_response, request
import librosa
import soundfile as sf
import wave
from .model_utils import load_model, transcribe
import pandas as pd


lessonpage = Blueprint('lessonpage', __name__)

model_dir = './static/DS_models'

model = load_model(model_dir)

def try_eval(ele):
    try:
        return eval(ele)
    except:
        return ele


def get_wrong(label, result):
    #get the difference
    diff = difflib.ndiff(label, result)
    d = list(diff)
    
    #find the wrong and missing phone
    wrong = []
    missing = []
    for idx,evaluate in enumerate(d):
        if "+" in evaluate or "-" in evaluate:
            if "+" in evaluate:
                wrong.append(idx)
            else:
                missing.append(idx)
    return wrong, missing


@lessonpage.route("/video/<video_id>")
def display(video_id):
    # video_id = eval(video_id)
    
    with open('./static/talks_for_app/{}/final_transcript.csv'.format(video_id), 'r', encoding='utf8') as f:
        reader = csv.reader(f)
        sentence_data = list(reader)[1:]
    
    sentence_data = [[try_eval(element) for element in sentence] for sentence in sentence_data]
    
    def get_sentences(offset=0, per_page=10):
        return sentence_data[offset: offset + per_page]
    
    page, per_page, offset = get_page_args(page_parameter='page',
                                           per_page_parameter='per_page')
    
    total = len(sentence_data)
    
    pagination_sentences = get_sentences(offset=offset, per_page=per_page)
    pagination = Pagination(page=page, per_page=per_page, total=total,
                            css_framework='bootstrap4')
    
    return render_template('lesson.html', 
                            video_id=video_id,
                            sentence_data=pagination_sentences,
                            page=page,
                            per_page=per_page,
                            pagination=pagination,
                            total=total
                           )


@lessonpage.route("/process_voice", methods=['POST'])
def process_voice():
    # req = request.get_json()
    print("In process voice function")
    print(request.headers)
    file_path = './uploads/file.wav'
    final_path = '../demo_app/uploads/file_final.wav'
    
    # print(io.BytesIO(request.data))
    #write file wav
    f = open(file_path, 'wb')
    f.write(request.data)
    
    f.close()

    au,_ = librosa.load(file_path, sr=48000)

    sf.write(final_path, au, 48000)

    result = transcribe(final_path, model)

    res = jsonify({'message':result})

    return res


@lessonpage.route("/search",  methods=['POST'])
def handle_search():
    # video_id = eval(video_id)
    keyword = request.form['keyword']

    words_df = pd.read_csv("./static/sentences/word_sent_id.csv")
    sent_id = eval(words_df[words_df['word'] == keyword]['sentence_id'].to_list()[0])

    sent_df = pd.read_csv("../data/sentences_tatoeba/eng_sentences_audio.csv", index_col=0)
    texts = sent_df.loc[sent_id]['text'].to_list()

    result = {idx:text for idx, text in zip(sent_id, texts)}
    
    def get_sentences(offset=0, per_page=10):
        return sent_id[offset: offset + per_page]
    
    page, per_page, offset = get_page_args(page_parameter='page',
                                           per_page_parameter='per_page')
    
    total = len(sent_id)
    
    pagination_sentences = get_sentences(offset=offset, per_page=per_page)
    pagination = Pagination(page=page, per_page=per_page, total=total,
                            css_framework='bootstrap4')


    # sentence_data = [[try_eval(element) for element in sentence] for sentence in sentence_data]
    
    return render_template('search_display.html', 
                        keyword=keyword, 
                        sent_id=sent_id, 
                        result=result,
                        sentence_data=pagination_sentences,
                        page=page,
                        per_page=per_page,
                        pagination=pagination,
                        total=total)