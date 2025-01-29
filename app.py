from pydoc import render_doc
from django.shortcuts import render
from flask import Flask, render_template, request, jsonify, Response
import pandas as pd
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm
from sympy import N
from csv import writer
import spotipy
import spotipy.util as util
from spotipy import SpotifyClientCredentials
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
import tensorflow as tf

tf.compat.v1.disable_eager_execution()
tf.compat.v1.disable_v2_behavior()

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Model,load_model
from keras_preprocessing.image import ImageDataGenerator , img_to_array, load_img


import cv2
import datetime, time
from threading import Thread

import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__, static_url_path='/static')
  
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/musicplayer', methods=['GET','POST'])
def musicplayer():
    return render_template('musicplayer.html')
  
@app.route('/refresh', methods=['GET','POST'])
def refresh():
    a = "87497b1a09094aa29244bcd9b4c3e843"
    b = "3a270da264f54cd19fe0c681d8232031"
    CLIENT_ID = a
    CLIENT_SECRET = b

    #Authentication - without user
    client_credentials_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
    sp = spotipy.Spotify(client_credentials_manager = client_credentials_manager)


    with open("./static/song.json", "w+") as outfile:
                json.dump("", outfile, indent = 4)
    data = pd.read_csv("Dataset.csv")
    x = data.iloc[:,3:14] 
    y=data['Class']
    max_accuracy=0
    
    X_train, X_test, y_train, y_test = train_test_split(x, y,test_size=0.3, random_state = 9)
    abc =AdaBoostClassifier(n_estimators=16,learning_rate=0.97)
    
    model = abc.fit(X_train, y_train)
    
    X_train, X_test, y_train, y_test = train_test_split(x, y,test_size=0.3, random_state = 291)
    xgb_cl = xgb.XGBClassifier()
    xgb_cl.fit(X_train, y_train)

    X_train, X_test, y_train, y_test = train_test_split(x, y,test_size=0.3, random_state = 18661)

    gnb = GaussianNB().fit(X_train, y_train)
    gnb_predictions = gnb.predict(X_test)
    
    
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state = 76)

    dtree_model = DecisionTreeClassifier(max_depth = 3).fit(X_train, y_train)

    max_i=0
    max_accuracy=0
    max_finalpred=[]
    for i in range(2):
        X_train, X_test, y_train, y_test = train_test_split(x, y,test_size=0.3, random_state = 1991)
        pred1=dtree_model.predict(X_test)
        pred2=gnb.predict(X_test)
        pred3=model.predict(X_test)
        pred4=xgb_cl.predict(X_test)
        finalpred=(1.501*pred4+0.499*pred1)/2
        for j in range(len(finalpred)):
            if(finalpred[j]<0.5):
                finalpred[j]=0
            elif((finalpred[j]>=0.5)&(finalpred[j]<1.5)):
                finalpred[j]=1
            elif((finalpred[j]>=1.5)&(finalpred[j]<2.5)):
                finalpred[j]=2
            else:
                finalpred[j]=3
        accuracy=accuracy_score(y_test,finalpred)
        if(accuracy>max_accuracy):
            max_accuracy=accuracy
            max_finalpred=finalpred

    return render_template('userinput.html')
    
@app.route('/mood', methods=['GET','POST'])
def userinputmap():

    with open("./static/song.json", "w+") as outfile:
                json.dump("", outfile, indent = 4)
    model=load_model("imagerec.h5")
    op={0: 'ANGRY', 3: 'HAPPY', 2: 'CALM', 1: 'SAD'}
    path = "image.png"
    
    img = load_img(path, target_size=(224,224) )
    
    emo = " "
    i = img_to_array(img)/255
    input_arr = np.array([i])
    input_arr.shape
    print(model.predict(input_arr))
    pred = (np.argmax(model.predict(input_arr)))
    print(pred)
    if(pred == 0):
        emo = "ANGRY"
    elif(pred == 1):
        emo = "SAD"
    elif(pred == 2):
        emo = "CALM"
    elif(pred == 3):
        emo = "HAPPY"
    
    print(f" the person is {emo}")
    songs_json = {"song_id_image":[]}
    dataset = pd.read_csv('song_id.csv',encoding="utf-8")
    dataframe = dataset.iloc[:,:]
    emotion = ""
    
    id = {}
    
    if request.method == 'POST':

        userinput = emo
        
        if userinput == 'CALM':
            
            for i in range(len(dataframe)) :
                flag = 1
                if dataframe.iloc[i,5] == "0":
                        id["Name"] = dataframe.iloc[i,0]
                        id["Artist"] = dataframe.iloc[i,1]
                        id["uri"] = dataframe.iloc[i,2]
                        id["Track"] = dataframe.iloc[i,3]
                        id["id"] = dataframe.iloc[i,4]
                        if(len(songs_json)>=1):
                            k = 0
                            for j in songs_json.values():
                                for k in j:
                                    if dataframe.iloc[i,4] != k['id']:
                                        continue
                                    else:
                                        flag = 0
                                        break
                            if flag == 1:
                                songs_json["song_id_image"].append(id)
                        else:
                            songs_json["song_id_image"].append(id)
                        id = {}
            with open("./static/song.json", "w+") as outfile:
                json.dump(songs_json, outfile, indent = 4)

        elif userinput == 'HAPPY':
            
            for i in range(len(dataframe)) :
                flag = 1
                if dataframe.iloc[i,5] == "1":
                        id["Name"] = dataframe.iloc[i,0]
                        id["Artist"] = dataframe.iloc[i,1]
                        id["uri"] = dataframe.iloc[i,2]
                        id["Track"] = dataframe.iloc[i,3]
                        id["id"] = dataframe.iloc[i,4]
                        if(len(songs_json)>=1):
                            k = 0
                            for j in songs_json.values():
                                for k in j:
                                    if dataframe.iloc[i,4] != k['id']:
                                        continue
                                    else:
                                        flag = 0
                                        break
                            if flag == 1:
                                songs_json["song_id_image"].append(id)
                        else:
                            songs_json["song_id_image"].append(id)
                        id = {}
            with open("./static/song.json", "w+") as outfile:
                json.dump(songs_json, outfile, indent = 4)

        
        elif userinput == 'SAD':
            for i in range(len(dataframe)) :
                flag = 1
                if dataframe.iloc[i,5] == "2":
                        id["Name"] = dataframe.iloc[i,0]
                        id["Artist"] = dataframe.iloc[i,1]
                        id["uri"] = dataframe.iloc[i,2]
                        id["Track"] = dataframe.iloc[i,3]
                        id["id"] = dataframe.iloc[i,4]
                        if(len(songs_json)>=1):
                            k = 0
                            for j in songs_json.values():
                                for k in j:
                                    if dataframe.iloc[i,4] != k['id']:
                                        continue
                                    else:
                                        flag = 0
                                        break
                            if flag == 1:
                                songs_json["song_id_image"].append(id)
                        else:
                            songs_json["song_id_image"].append(id)
                        id = {}
            with open("./static/song.json", "w+") as outfile:
                json.dump(songs_json, outfile, indent = 4)
            

        elif userinput == 'ANGRY':
            for i in range(len(dataframe)) :
                flag = 1
                if dataframe.iloc[i,5] == "3":
                        id["Name"] = dataframe.iloc[i,0]
                        id["Artist"] = dataframe.iloc[i,1]
                        id["uri"] = dataframe.iloc[i,2]
                        id["Track"] = dataframe.iloc[i,3]
                        id["id"] = dataframe.iloc[i,4]
                        if(len(songs_json)>=1):
                            k = 0
                            for j in songs_json.values():
                                for k in j:
                                    if dataframe.iloc[i,4] != k['id']:
                                        continue
                                    else:
                                        flag = 0
                                        break
                            if flag == 1:
                                songs_json["song_id_image"].append(id)
                        else:
                            songs_json["song_id_image"].append(id)
                        id = {}
            with open("./static/song.json", "w+") as outfile:
                json.dump(songs_json, outfile, indent = 4)
        emotion+=op[pred]
    
    return render_template('userinput.html', e = emotion)
    

global capture,rec_frame,grey, switch, neg, face, rec, out 
capture=0
grey=0
neg=0
face=0
switch=1
rec=0

net = cv2.dnn.readNetFromCaffe('./saved_model/deploy.prototxt.txt', './saved_model/res10_300x300_ssd_iter_140000.caffemodel')

camera = cv2.VideoCapture(0)

def record(out):
    global rec_frame
    while(rec):
        time.sleep(0.05)
        out.write(rec_frame)

def detect_face(frame):
    global net
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0))   
    net.setInput(blob)
    detections = net.forward()
    confidence = detections[0, 0, 0, 2]

    if confidence < 0.5:            
            return frame           

    box = detections[0, 0, 0, 3:7] * np.array([w, h, w, h])
    (startX, startY, endX, endY) = box.astype("int")
    try:
        frame=frame[startY:endY, startX:endX]
        (h, w) = frame.shape[:2]
        r = 480 / float(h)
        dim = ( int(w * r), 480)
        frame=cv2.resize(frame,dim)
    except Exception as e:
        pass
    return frame
 

def gen_frames():  
    global out, capture,rec_frame
    while True:
        success, frame = camera.read() 
        if success:
            if(face):                
                frame= detect_face(frame)
            if(capture):
                capture=0
                now = datetime.datetime.now()
                cv2.imwrite("image.png", frame)
                cv2.imshow("",frame)
                camera.release()
            
            if(rec):
                rec_frame=frame
                frame= cv2.putText(cv2.flip(frame,1),"Recording...", (0,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),4)
                frame=cv2.flip(frame,1)
            
                
            try:
                ret, buffer = cv2.imencode('.jpg', cv2.flip(frame,1))
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                pass
                
        else:
            pass

        
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/requests',methods=['POST','GET'])
def tasks():
    global switch,camera
    if request.method == 'POST':
        if request.form.get('click') == 'CAPTURE':
            global capture
            capture=1                
                 
    elif request.method=='GET':
        return render_template('userinput.html')
    return render_template('userinput.html')


if __name__ == '__main__':
    app.run()
    
    
camera.release()
cv2.destroyAllWindows()     