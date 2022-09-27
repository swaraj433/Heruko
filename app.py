import joblib
import numpy as np
import pandas as pd
from flask import Flask,request,jsonify

app = Flask(__name__)
model = joblib.load('randomforest.pkl') 

@app.route('/')
def home():
    return "Hello World"


@app.route('/predict', methods=['POST'])
def predict():
        f_list = [request.form.get('limit'), request.form.get('education'), request.form.get('marriage'),
                  request.form.get('pay_delay1'),
                  request.form.get('pay_delay2'), request.form.get('pay_delay3'),request.form.get('bill_amt1'),
                  request.form.get('bill_amt2'),request.form.get('bill_amt3'),request.form.get('bill_paid1'),
                  request.form.get('bill_paid2')]  # list of inputs

        final_features = np.array(f_list).reshape(-1, 11)
        df = pd.DataFrame(final_features)

        prediction = model.predict(df)
        result = prediction[0]

        return jsonify({'prediction':str(result)})


if __name__ == "__main__":
    app.run(debug=True)
