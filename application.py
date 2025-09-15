from flask import Flask,render_template,url_for,request
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle

application=Flask(__name__)
app=application

## models
ridge_model=pickle.load(open('models/ridge.pkl','rb'))
scale=pickle.load(open('models/scale.pkl','rb'))


@app.route('/',methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/prediction',methods=['GET','POST'])
def prediction():
    if request.method=='POST':
        temp=float(request.form["temparature"])
        RH=float(request.form["RH"])
        WS=float(request.form["WS"])
        Rain=float(request.form["Rain"])
        FFMC=float(request.form["FFMC"])
        DMC=float(request.form["DMC"])
        ISI=float(request.form["ISI"])
        Classes=float(request.form['Classes'])
        Region=float(request.form["Region"])
        
        scaled=scale.transform([[temp,RH,WS,Rain,FFMC,DMC,ISI,Classes,Region]])
        pred_value=ridge_model.predict(scaled)
        return render_template('home.html',result=pred_value[0])
    else:
        return render_template('home.html')



if __name__=='__main__':
    app.run(host='0.0.0.0')
