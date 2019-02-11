from frozen_model import BertQA
import argparse
import io
import os
import json
import flask
from flask import redirect, url_for, request, render_template, flash, send_file

app = application = flask.Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'GET':
        # show the upload form
        return render_template('index.html')

    if request.method == 'POST':
        # check if a file was passed into the POST request
        if 'qas' not in request.files:
            flash('No file was uploaded.')
            return redirect(request.url)

        json_file = request.files['qas']

        # if filename is empty, then assume no upload
        if json_file.filename == '':
            flash('No file was uploaded.')
            return redirect(request.url)

        # if the file is "legit"
        if json_file:
            try:
                filename = json_file.filename
                filepath = os.path.join('./uploads', filename)
                if not os.path.exists('uploads'):
                    os.mkdir('uploads')

                json_file.save(filepath)
                passed = True
            except ValueError:
                passed = False

            if passed:
                return redirect(url_for('predict', filename=filename))
            else:
                flash('An error occurred, try again.')
                return redirect(request.url)


@app.route('/predict/<filename>', methods=['GET'])
def predict(filename):
    # TODO: Logic to load the uploaded image filename and predict the
    filepath = os.path.join('./uploads', filename)

    json_file = bert_model.process(filepath)

    with io.open('./uploads/debug.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(json_file, ensure_ascii=False))

    json_url = url_for('qas', filename="debug.json")
    return render_template(
        'predict.html',
        json_url=json_url
    )


@app.errorhandler(500)
def server_error(error):
    return render_template('error.html'), 500


@app.route('/qas/<filename>', methods=['GET'])
def qas(filename):
    return send_file(os.path.join('uploads', filename))


@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run API')
    parser.add_argument('cfg_path', help='Configuration file path')
    parser.add_argument('port', help='Opening port of service to receive message from client')
    args = parser.parse_args()

    global bert_model
    bert_model = BertQA(args.cfg_path)
    app.run(port=int(args.port), host="0.0.0.0")
