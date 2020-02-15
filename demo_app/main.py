from flask import Flask, render_template
from flask import jsonify, make_response, request

app = Flask(__name__)
from blueprints import *


# Disable cache
app.config['TEMPLATES_AUTO_RELOAD'] = True

app.register_blueprint(homepage)
app.register_blueprint(lessonpage)


if __name__ == '__main__':
  app.run(host='127.0.0.1', port=8000, debug=True)