from flask import Blueprint, render_template
import csv

homepage = Blueprint('homepage', __name__)

def try_eval(ele):
    try:
        return eval(ele)
    except:
        return ele

@homepage.route('/', methods=['GET', 'POST'])
def index():

  with open('./static/talks_for_app/talks_data.csv', 'r', encoding='utf8') as f:
      reader = csv.reader(f)
      talks = list(reader)[1:]
  
  talks = [[try_eval(element) for element in talk] for talk in talks]

  return render_template('home.html', video_data = talks)