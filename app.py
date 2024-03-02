import pandas as pd
import numpy as np
import pickle

from flask import Flask,jsonify,render_template,request

app=Flask(__name__)
reg=pickle.load(open('RandomForestModel.pkl','rb'))
encoder=pickle.load(open('encoder.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    data = request.json['data']
    data = pd.DataFrame(data, index=[0])
    output = reg.predict(np.array(data.values))
    
    # Convert the output to a JSON-serializable format
    json_output = {'prediction': output[0].item()}  # Convert NumPy int32 to Python int
    
    return jsonify(json_output)


if __name__=="__main__":
    app.run(debug=True)