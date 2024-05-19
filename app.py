from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Load the model
with open('credit_score_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route("/")
def Home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()
    data = {key: float(value) if value else 0 for key, value in data.items()}
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    return render_template('index.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
