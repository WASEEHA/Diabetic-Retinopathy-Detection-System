from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import os
import pandas as pd
import tensorflow as tf
import cv2
from keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
from io import BytesIO

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

app = Flask(__name__, static_url_path='/static')

isLoggedIn = False

global model
model = load_model('C:/Users/wasee/OneDrive/Desktop/DRD/Diabetic-Retinopathy-Detection-System-master/model_2.h5')
class_labels = ['Mild', 'Moderate', 'No_DR', 'Proliferate_DR', 'Severe']

def preprocess_image(img, target_size=(224,224)):
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_YCrCb = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2YCrCb)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16,16))
    cl1 = clahe.apply(img_YCrCb[:,:,0])
    img_YCrCb[:,:,0] = cl1         
    img_RGB_2 = cv2.cvtColor(img_YCrCb, cv2.COLOR_YCrCb2RGB)
    img_blur = cv2.GaussianBlur(img_RGB_2, (5, 5), 0)
    img_resized = cv2.resize(img_blur,target_size)
    return img_resized

data = "C:/Users/wasee/OneDrive/Desktop/DRD/Diabetic-Retinopathy-Detection-System-master/dataset"
train_dir = "C:/Users/wasee/OneDrive/Desktop/DRD/Diabetic-Retinopathy-Detection-System-master/dataset/train"
classes = sorted(os.listdir(train_dir))
x_train = np.array([preprocess_image(cv2.imread(os.path.join(train_dir, cl, name), cv2.IMREAD_COLOR)) for cl in classes for name in os.listdir(os.path.join(train_dir, cl))])

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    validation_split=0.2
)
datagen.fit(x_train)

def predict_class(image_path):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        preprocessed_image = preprocess_image(img)
        preprocessed_image = np.expand_dims(preprocessed_image, axis=0)
        x_test = np.array(preprocessed_image)
        x_test = (x_test - datagen.mean) / (datagen.std + 0.000001)
        predictions = model.predict(x_test)
        predicted_class_index = predictions.argmax(axis=-1)[0]
        predicted_class = class_labels[predicted_class_index]
        return predicted_class
    except Exception as e:
        print("Error occurred during prediction:", e)
        return None


@app.route('/')
def index():
    return render_template('login.html', message="Please log in.")

@app.route('/login', methods=['POST'])
def login():
    global isLoggedIn
    username = request.form['username']
    password = request.form['password']

    if username == "Technician" and password == "abcd@123":
        print("Login successful")
        isLoggedIn = True
        return redirect(url_for('welcome'))
    else:
        print("Invalid credentials")
        return render_template('login.html', message="Invalid username or password. Please try again.")

@app.route('/welcome')
def welcome():
    if isLoggedIn:
        return render_template('welcome.html')
    return "Unauthorized", 401

@app.route('/home')
def home():
    print("Rendering home.html")
    return render_template('home.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files or request.files['image'].filename == '':
        return render_template('error.html')

    image = request.files['image']
    filename = secure_filename(image.filename)
    image_path = os.path.join(train_dir, 'Severe', filename)
    image.save(image_path)
    return redirect(url_for('wait', image_path=image_path))
    

@app.route('/wait')
def wait():
    image_path = request.args.get('image_path')
    if image_path:
        print("Wait route")
    image_path = request.args.get('image_path')
    print("Image path:", image_path)
    predicted_class = predict_class(image_path)
    print("Predicted class:", predicted_class)
    
    results_csv = 'predictions.csv'
    if not os.path.isfile(results_csv):
        results_df = pd.DataFrame(columns=['image_name', 'predicted_outcome'])
    else:
        results_df = pd.read_csv(results_csv)
    
    image_filename = os.path.basename(image_path)
    new_data = {'image_name': [image_filename], 'predicted_outcome': [predicted_class]}
    results_df = pd.concat([results_df, pd.DataFrame(new_data)], ignore_index=True)
    
    results_df.to_csv(results_csv,index= False)
    predicted_class = predict_class(image_path)
    if predicted_class == "Mild":
            return render_template('output1.html')
    elif predicted_class == "Moderate":
            return render_template('output2.html')
    elif predicted_class == "No_DR":
            return render_template('output3.html')
    elif predicted_class == "Proliferate_DR":
            return render_template('output4.html')
    elif predicted_class == "Severe":
            return render_template('output5.html')
    elif predicted_class == None:
            return render_template('output6.html')
    else:
        return "Image path not provided", 400

@app.route('/logout')
def logout():
    global isLoggedIn
    isLoggedIn = False
    return redirect(url_for('index'))

@app.route('/about')
def about():
    return render_template('about.html')
    
@app.route('/contact')
def contact():
    return render_template('contact.html')


@app.route('/statistics')
def statistics():
    pie_chart_path = 'static/pie_chart.png'
    return render_template('statistics.html', pie_chart=pie_chart_path)


if __name__ == '__main__':
    app.run(debug=True)
