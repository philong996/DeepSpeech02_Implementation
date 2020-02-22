from flask import Blueprint, render_template
import pandas as pd
import csv

homepage = Blueprint('homepage', __name__)

def try_eval(ele):
    try:
        return eval(ele)
    except:
        return ele

@homepage.route('/', methods=['GET', 'POST'])
def index():

    sent_df = pd.read_csv("../data/sentences_tatoeba/final_eng_sentences_audio.csv", index_col=0)
  
    result = sent_df.sample(n=20)['text'].to_dict()
    
    sent_id = list(result.keys())

    return render_template('home.html', sent_id=sent_id ,result = result)