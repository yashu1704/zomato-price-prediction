from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load('zomato_price_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    restaurant_type = request.form['restaurant_type']
    avg_cost = float(request.form['avg_cost'])
    num_ratings = int(request.form['num_ratings'])
    online_order = request.form['online_order']
    table_booking = request.form['table_booking']
    
    input_data = pd.DataFrame({
        'avg cost (two people)': [avg_cost],
        'num of ratings': [num_ratings],
        'online_order_Yes': [1 if online_order == 'Yes' else 0],
        'table booking_Yes': [1 if table_booking == 'Yes' else 0],
        'restaurant type': [restaurant_type]
    })

    prediction = model.predict(input_data)

    return render_template('result.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
