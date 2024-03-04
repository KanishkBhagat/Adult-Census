import pandas as pd
import numpy as np
import pickle
import json
from flask import Flask,render_template,jsonify,request,url_for,app

app=Flask(__name__)
reg=pickle.load(open('RandomForestModel.pkl','rb'))
encoder=pickle.load(open('encoder.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')
    
# @app.route('/predict_api',methods=['POST'])
# def predict_api():
#     data=request.json['data']
#     data=pd.DataFrame(data,index=[0])
#     #data=encoder.fit_transform(data.values)
#     output=reg.predict(data.values)
#     json_data={'prediction':output[0].item()}
#     return jsonify(json_data)

@app.route('/predictdata',methods=['GET','POST'])
def predictdata():
    data=[float(x) for x in request.form.values()]
    data=[np.array(data)]
    pred=reg.predict(data)
    return render_template('home.html',prediction_text='prediction:{}'.format(pred))
    
if __name__=="__main__":
    app.run(debug=True)


    
    