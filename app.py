
from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np

app = Flask(__name__)

test=pickle.load(open('test1.pkl','rb'))


@app.route('/')
def hello_world():
    return render_template("t.html")

@app.route('/predict',methods=['POST','GET'])
def predict():
    int_features=[int(x) for x in request.form.values()]
    final=[np.array(int_features)]
    print(int_features)
    print(final)
    prediction=test.predict(final)

    if prediction == 0:
        return render_template('t.html',pred="\t\t\t\t\tProbability of accident severity is : Minor")
    else:
        return render_template('t.html',pred="\t\t\t\t\tProbability of accident severity is : Major")

@app.route('/Map')
def map1():
    return render_template("map.html")    

@app.route('/Graphs')
def graph():
    return render_template("graph.html")

@app.route('/Map1')
def map2():
    return render_template("ur.html")

@app.route('/Map2')
def map3():
    return render_template("bs.html")

@app.route('/Map3')
def map4():
    return render_template("hm.html")

@app.route('/Pie')
def pie():
    return render_template("pie.html")


if __name__=="__main__":
    app.run() 
