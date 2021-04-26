from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np
import preprocessing
from preprocessing import pipeline_preprocessing2
#import model_rfc
#from model_rfc import cleaner

app = Flask(__name__)


#file_IE = 'rfc_IE.pkl'
#file_JP = 'rfc_JP.pkl'
#file_NS = 'rfc_NS.pkl'
#file_TF = 'rfc_TF.pkl'
#file_pipeline='pipeline.pkl'

load_model_rfc_IE = pickle.load(open('rfc_IE.pkl', 'rb'))
load_model_rfc_JP = pickle.load(open('rfc_JP.pkl', 'rb'))
load_model_rfc_NS = pickle.load(open('rfc_NS.pkl', 'rb'))
load_model_rfc_TF = pickle.load(open('rfc_TF.pkl', 'rb'))
#load_pipeline = pickle.load(open('pipeline.pkl','rb'))


@app.route('/')
def hello_world():
    return render_template("index.html")


@app.route('/predict',methods=['POST','GET'])
def output():
    content=request.form['posts']
    
    str=[content]

    test_input = pipeline_preprocessing2.fit_transform(str)
    IE=load_model_rfc_IE.predict(test_input)[0]
    JP=load_model_rfc_JP.predict(test_input)[0]
    NS=load_model_rfc_NS.predict(test_input)[0]
    TF=load_model_rfc_TF.predict(test_input)[0]

    res= IE+" "+NS+" "+TF+" "+" "+JP
    
    return render_template('index.html',pred='Your Personality is {}'.format(res))
    


if __name__ == '__main__':
    app.run(debug=True)
