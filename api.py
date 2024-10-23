from flask import Flask,redirect,url_for,render_template,request
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

@app.route('/', methods=['GET','POST'])
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    prg = int(request.form['prg'])
    glc = int(request.form['gl'])
    bp = int(request.form['bp'])
    skt = int(request.form['sk'])
    ins = int(request.form['ins'])  
    bmi = float(request.form['BMI'])
    dpf = float(request.form['ped'])
    age = int(request.form['age'])


    final_features = np.array([(prg,glc,bp,skt,ins,bmi,dpf,age)])
    prediction = model.predict(final_features)
    return render_template('index.html',prediction_text='THE PATIENT HAS DIABETES : {}'.format(prediction))

if __name__=="__main__":
    app.run(debug=True)
