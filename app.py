from flask import Flask,request
from saved_model.predict import predict_data
import json

# create the flask object
app = Flask(__name__)

@app.route('/')
def index():
    return "Index Page"

@app.route('/predict',methods=['GET','POST'])
def predict():
    data = request.form.get('data')
    #print('data: {}'.format(data))
    if data == None:
        return 'Got None'
    else:
        # model.predict.predict returns a dictionary
        prediction = predict_data(data) 
    return json.dumps(str(prediction))

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)