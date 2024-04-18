## Importing all necessary packages and libraries
from flask import Flask, render_template, request
import numpy as np
import pickle
import warnings
import psycopg2
import config

app = Flask(__name__)
#app.static_folder = 'static'
loaded_model = pickle.load(open("model.pkl", 'rb'))

warnings.filterwarnings('ignore')

con = psycopg2.connect(host=config.host, dbname=config.database, user=config.username, password=config.password,
                       port=config.port)
# con=psycopg2.connect(host=config.host,dbname=config.database,user=config.username,password=config.password,port=config.port)
cur = con.cursor()

con.commit()
cur.close()
con.close()

@app.route('/')
def home():
    return render_template('Home.html')


@app.route('/predict', methods=['POST'])
def predict():
    con = psycopg2.connect(host=config.host, dbname=config.database, user=config.username, password=config.password,
                           port=config.port)
    cur = con.cursor()
    N = int(request.form['Nitrogen'])
    P = int(request.form['Phosporus'])
    K = int(request.form['Potassium'])
    temp = float(request.form['Temperature'])
    humidity = float(request.form['Humidity'])
    ph = float(request.form['pH'])
    rainfall = float(request.form['Rainfall'])

    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1, -1)

    prediction = loaded_model.predict(single_pred)

    crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                 14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                 19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}

    if prediction[0] in crop_dict:
        crop = crop_dict[prediction[0]]
        result = "{} is the best crop to be cultivated right there".format(crop)
    else:
        result = "Sorry, we could not determine the best crop to be cultivated with the provided data."


    #insert_script = "INSERT INTO CropPredict (Nitrogen, Phosphorus,Potassium,Temperature,Humidity,pH,Rainfall) values (" \
                    #"%s,%s,%s,%s,%s,%s,%s) "
    #value = ('0','Trilochan Sahu', str(feature_list), int(prediction[0]))
    cur.execute("INSERT INTO croppredict (Nitrogen, Phosphorus, Potassium, Temperature, Humidity, pH, Rainfall, predicted_crop) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",(N, P, K, temp, humidity, ph, rainfall, crop))
    con.commit()
    cur.close()
    con.close()
    return render_template('Home.html', prediction=result)
@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/submit_contact', methods=['POST'])
def submit_contact():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        message = request.form['message']

        # You can add code here to handle the contact form submission, such as sending an email to the team.

        return "Thank you for your message! We will get back to you shortly."

if __name__ == '__main__':
    app.run(debug=True)