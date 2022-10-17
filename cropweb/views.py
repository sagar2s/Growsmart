import pickle
from django.shortcuts import render
from flask import Markup
from matplotlib.style import context
from extra.disease import disease_dic
# Create your views here.
from django.shortcuts import render, redirect
from django.http import HttpResponseRedirect, HttpResponse
from django.views.generic import View
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.contrib.auth import authenticate, logout, login
from django.http import Http404
from django.core.files.storage import FileSystemStorage
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow_hub as hub
from keras.models import load_model
import sklearn
from django import template
register = template.Library()


with open('./saved_model/RandomForest.pkl', 'rb') as f:
    crop_model = pickle.load(f)
# Plant leaves classes 38 total
class_names = ['Apple___Apple_scab',
 'Apple___Black_rot',
 'Apple___Cedar_apple_rust',
 'Apple___healthy',
 'Blueberry___healthy',
 'Cherry_(including_sour)___Powdery_mildew',
 'Cherry_(including_sour)___healthy',
 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
 'Corn_(maize)___Common_rust_',
 'Corn_(maize)___Northern_Leaf_Blight',
 'Corn_(maize)___healthy',
 'Grape___Black_rot',
 'Grape___Esca_(Black_Measles)',
 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
 'Grape___healthy',
 'Orange___Haunglongbing_(Citrus_greening)',
 'Peach___Bacterial_spot',
 'Peach___healthy',
 'Pepper,_bell___Bacterial_spot',
 'Pepper,_bell___healthy',
 'Potato___Early_blight',
 'Potato___Late_blight',
 'Potato___healthy',
 'Raspberry___healthy',
 'Soybean___healthy',
 'Squash___Powdery_mildew',
 'Strawberry___Leaf_scorch',
 'Strawberry___healthy',
 'Tomato___Bacterial_spot',
 'Tomato___Early_blight',
 'Tomato___Late_blight',
 'Tomato___Leaf_Mold',
 'Tomato___Septoria_leaf_spot',
 'Tomato___Spider_mites Two-spotted_spider_mite',
 'Tomato___Target_Spot',
 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 'Tomato___Tomato_mosaic_virus',
 'Tomato___healthy']
supplement_info = pd.read_csv('supplement_info.csv',encoding='cp1252')
disease_info = pd.read_csv('disease_info.csv' , encoding='cp1252')

def index(request):
    return render(request, 'index.html')

def weather_fetch(url):
    pass
#Crop Recommendation
def recommend(value):
    return
def crop_predict(request):
    if request.method == 'POST':
        N = int(request.POST['nitrogen'])
        print(N)
        P = int(request.POST['phosphorous'])
        K = int(request.POST['pottasium'])
        temperature = float(request.POST['Temperature'])
        humidity = float(request.POST['humidity'])
        ph = float(request.POST['ph'])
        rainfall = float(request.POST['rainfall'])
        data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        my_prediction = crop_model.predict(data)
        final_prediction = my_prediction[0]
        context={'final':final_prediction,}
        return render(request, 'crop-result.html',context=context)

    return render(request, 'crop.html')
 


def crop_recommend(request):
    title = 'Grow Smart: Crop Recommendation'
    return render(request, 'crop_recom.html')

def geo_coming(request):
    title = 'Coming soon'
    return render(request, 'coming_soon.html')
# Plant Disease detection sector
def predict_image(url):
    model = load_model(r'.\saved_model\detect_using_cnn.h5')
    path = url
    image = cv2.imread(path)
    image = np.array(
        Image.open(path).convert("RGB").resize((224, 224)) # image resizing
    )

    image = image/255 # normalize the image in 0 to 1 range
    img_array = tf.expand_dims(image, 0)
    predictions = model.predict(img_array)

    print("Predictions:",predictions)

    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    print("class:", predicted_class)
    print("confidence:", confidence)
    return predicted_class
def disease_detect(request):
    title="Grow Smart: Disease Detection"
    if request.method == 'POST':
        if 'image' not in request.FILES:
            return HttpResponseRedirect(request.path_info)
        file = request.FILES['image']
        print(file)
        fs = FileSystemStorage()
        a = fs.save(file.name,file)
        path=fs.url(a)
        img='.'+path
        print(img)
        prediction = predict_image(img)
        prediction = Markup(str(disease_dic[prediction]))
        context={'prediction' : prediction,
                'MOD':img}
        return render(request, 'disease-result.html', context)
    return render(request,'disease.html')
def fertstore(request):
    supplement_image = list(supplement_info['supplement image'])
    supplement_name = list(supplement_info['supplement name'])
    supplement_index=list(supplement_info['index'])
    disease = list(disease_info['disease_name'])
    buy = list(supplement_info['buy link'])
    context={'supplement_image': supplement_image,
            'supplement_name': supplement_name,
            'lengt': supplement_index,
            'disease': disease,
            'buy': buy}
    return render(request,'store.html', context)

