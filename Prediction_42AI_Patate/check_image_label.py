#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from flask import Flask, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from flask import send_from_directory
import cv2
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from math import ceil
from pathlib import Path

# In command line, type:
# $ FLASK_APP=check_image_label.py flask run
# or :
# $ python check_image_label.py


# Create an 'upload' directory if it does not already exist
Path('static/uploads').mkdir(parents=True, exist_ok=True)

# print(os.getcwd()) # check if we are in the correct working directory 'Prediction_Web/'
UPLOAD_FOLDER = os.path.join('static', 'uploads')
ALLOWED_EXTENSIONS_img = set(['png', 'jpg', 'jpeg', 'gif'])
ALLOWED_EXTENSIONS_json = set(['json'])
ALLOWED_EXTENSIONS_model = set(['ckpt', 'h5'])

# Define Flask app with customizable static_folder
app = Flask(__name__, static_folder='static')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# Photos: Verify if filename is allowed
def allowed_file_img(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS_img

# json: Verify if filename is allowed
def allowed_file_json(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS_json
           
# Model: Verify if filename is allowed
def allowed_file_model(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS_model

def get_prediction(path_img, model_path, nb_dir):
    # convert image into tensor
    img = tf.convert_to_tensor(cv2.imread(path_img) / 255.0)
    img = tf.expand_dims(img, 0)
        
    # load model and get predictions
    model = load_model(model_path)
    predictions_raw = model(img)
    predicted_labels = [int(np.argmax(pred)) for pred in predictions_raw]
    motor_direction = predicted_labels[0]
    perc_direction = float(predictions_raw[0][0][motor_direction])
    center = ceil(int(nb_dir) / 2)
    if motor_direction == center:
        motor_speed = predicted_labels[1]
    else:
        motor_speed = 0
    return motor_direction, perc_direction, motor_speed


def get_true_labels(path_js, name):
    with open(path_js, 'r') as f:
        data=f.read()
        # parse file
    js = json.loads(data)
    true_direction = js[name]['label']['label_direction']
    true_speed = js[name]['label']['label_speed']
    return true_direction, true_speed

    
# Main Flask function: Get and Post data
@app.route('/', methods=['GET','POST'])
def upload_file():
    # Initialisation:
    # Output variables are set by default to -1 in case input data is missing.
    motor_direction = -1
    perc_direction = -1
    motor_speed = -1
    true_direction = -1
    true_speed = -1
    # Data variables are set by default to 0.
    photo = 0
    js_file = 0
    nb_dir = 0
    name = 0
    
    if request.method == 'POST':
        uploaded_img = request.files['im']
        uploaded_json = request.files['js']
        if uploaded_img:
            # get photo
            if allowed_file_img(uploaded_img.filename):
                photo = secure_filename(uploaded_img.filename)
                name = photo.rsplit('.', 1)[0]
                path_img = os.path.join(app.config['UPLOAD_FOLDER'], "img.jpg")
                uploaded_img.save(path_img)
        if uploaded_json:            
            # get json file
            if allowed_file_json(uploaded_json.filename):
                js_file = secure_filename(uploaded_json.filename)
                path_js = os.path.join(app.config['UPLOAD_FOLDER'], js_file)
                uploaded_json.save(path_js)
                
        # get model ### needs to be improved to check for errors
        try:
            model_path = request.form['model']
            nb_dir = request.form['nb_dir']
        except:
            print("Unable to find path. Please make sure it's valid and try again.")
  
        # get labels prediction
        if photo:
            motor_direction, perc_direction, motor_speed = get_prediction(path_img, model_path, nb_dir)
            
        # get true labels
        if js_file:
            true_direction, true_speed = get_true_labels(path_js, name)

        return render_template("result.html", pred_dir=motor_direction, pred_speed=motor_speed, true_dir=true_direction, true_speed=true_speed, perc_dir=round(perc_direction * 100, 2))
    return render_template("upload.html")

if __name__ == "__main__":
    app.run(debug=True)

