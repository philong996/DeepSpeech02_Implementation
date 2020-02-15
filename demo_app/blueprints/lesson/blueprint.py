from flask import Blueprint, render_template, Response
from flask_paginate import Pagination, get_page_args
import csv
import json
from flask import jsonify, make_response, request
import librosa
import soundfile as sf
import wave
from .model_utils import load_model, transcribe

lessonpage = Blueprint('lessonpage', __name__)

model_dir = '../DS_models'

model = load_model(model_dir)

def try_eval(ele):
    try:
        return eval(ele)
    except:
        return ele

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
    file_path = './uploads/file.wav'
    final_path = '../demo_app/uploads/file_final.wav'
    
    # print(io.BytesIO(request.data))
    #write file wav
    f = open(file_path, 'wb')
    f.write(request.data)
    
    f.close()

    au,sr = librosa.load(file_path, sr=48000)

    sf.write(final_path, au, 48000)

    result = transcribe(final_path, model)

    res = jsonify({'message':result})

    return res