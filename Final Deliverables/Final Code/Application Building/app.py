from distutils.log import debug
from flask import Flask, render_template, request
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import tensorflow as tf
import os
from werkzeug.utils import secure_filename
from tensorflow.python.keras.backend import set_session

app = Flask(__name__)
fruit_index=['Apple_Black_rot', 'Apple_healthy', 'Corn_(maize)_healthy', 'Corn_(maize)_Northern_Leaf_Blight', 'Peach_Bacterial_spot', 'Peach_healthy']
veg_index=['Pepper,_bell_Bacterial_spot','Pepper,_bell_healthy','Potato_Early_blight','Potato_healthy','Potato_Late_blight','Tomato_Bacterial_spot','Tomato_Late_blight','Tomato_Leaf_Mold','Tomato___Septoria_leaf_spot']

fruitmodel = load_model("fruit_training.h5")
vegmodel = load_model("veg_training.h5")

@app.route("/", methods=['GET'])
def home():
    return render_template('index.html')

@app.route("/login", methods=['GET'])
def login():
    return render_template('login.html')

@app.route("/predict", methods = ['POST'])
def predict():
    listofvalues = []

    if request.method == 'POST':
        file = request.files['leaf']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'static/uploads', secure_filename(file.filename))
        file.save(file_path)
        img = image.load_img(file_path, target_size = (128, 128))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis = 0)
        family = request.form['leaf-family']
        listofvalues.append('uploads/' + secure_filename(file.filename))

        if(family == 'vegetables'):
            preds=np.argmax(vegmodel.predict(x),axis=1)
            df = pd.read_excel('precautions - veg.xlsx')
            caution = df.iloc[preds[0]]['caution']
            disease = veg_index[preds[0]]
            listofvalues.append(disease)
            listofvalues.append(caution)

        else:
            preds = np.argmax(fruitmodel.predict(x),axis=1)
            df = pd.read_excel('precautions - fruits.xlsx')
            caution = df.iloc[preds[0]]['caution']
            disease = fruit_index[preds[0]]
            listofvalues.append(disease)
            listofvalues.append(caution)
        
        return render_template('predict.html',prediction=listofvalues)

if __name__ == "__main__":
    app.run(debug = True)
