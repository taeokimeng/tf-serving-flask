from flask import Flask, request, render_template, redirect, url_for, jsonify
import requests
import json
from serving_rest import request_prediction


app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    # Send a sentence by using JSON. Postman can be used for the test.
    try:
        inputs = json.loads(request.get_data())  # request.get_json()
    # JSON hasn't been sent.
    except json.decoder.JSONDecodeError:
        pass
    else:
        if len(inputs['input']) != 0:
            sentence = inputs['input']
            return jsonify({'message': 'SUCCESS', 'result': request_prediction(sentence)}, 201)

    if request.method == 'POST':
        input_string = request.form['sentence']
        output_string = request_prediction(input_string)
        return render_template('index.html', input=input_string, output=output_string)  # redirect(url_for('serving', sentence=input_string))

    return render_template('index.html')


@app.route('/serving/<sentence>')
def serving(sentence):
    output = request_prediction(sentence)
    return output


if __name__ == '__main__':
    app.debug = True
    app.run()
