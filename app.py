from flask import Flask, render_template, request
import pandas as pd 
import numpy as np 
from fcmeans import FCM
from sklearn.cluster import KMeans

app = Flask(__name__)

# Load the data
data = pd.read_csv("/Users/srinivasgollapalli/Desktop/crime.csv")
arr = data.Magnitude.unique()
arr = np.delete(arr, 7)
arr = arr.astype(int)

# Replace 'ARSON' with a numeric value or remove it
data['Magnitude'] = data['Magnitude'].replace('ARSON', '0')  # Replace 'ARSON' with '0' or any other suitable value

# Perform clustering
X = data[['Latitude', 'Longitude', 'Magnitude']]
X = X.astype(float)  # Ensure all values are float
my_model = FCM(n_clusters=3, random_state=42)
my_model.fit(X.values)  # Pass values instead of the DataFrame
fcm_clusters = my_model.predict(X.values)  # Predict using values
kmeans = KMeans(3, random_state=42, max_iter=20)
kmeans.fit(X.values)  # Pass values instead of the DataFrame
kmeans_clusters = kmeans.fit_predict(X.values)  # Predict using values

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    latit = float(request.form['latitude'])
    longit = float(request.form['longitude'])
    rate = 0.0
    c = 0
    for i in range(0, len(X)):
        if (X.iloc[i,0] == latit) & (X.iloc[i,1] == longit):  # Access DataFrame using iloc
            rate = X.iloc[i,2]
            c += 1
    safety = 100 - c
    return render_template('result.html', latitude=latit, longitude=longit, safety=safety)

if __name__ == '__main__':
    app.run(debug=True,port=5001)
