from flask import Flask, render_template, request
from pickle import load
import numpy as np
from sklearn.preprocessing import MinMaxScaler
# print(flask.__version__)

app = Flask(__name__)

# load the model
model = load(open('car_price_lr_model.pkl', 'rb'))
# load the scaler
x_scaler = load(open('x_scaler.pkl', 'rb'))
y_scaler = load(open('y_scaler.pkl', 'rb'))

@app.route('/', methods=["GET", "POST"])
def hello():
    predicted_price = []
    if request.method == 'POST':
        symboling = request.form.get("symboling")
        fueltype = request.form.get("fueltype")
        aspiration = request.form.get("aspiration")
        door = request.form.get("door")
        car_body_sedan = request.form.get("car_body_sedan")
        drivewheel_fwd = request.form.get("drivewheel_fwd")
        enginelocation = request.form.get("enginelocation")
        wheelbase = request.form.get("wheelbase")
        carlength = request.form.get("carlength")
        carwidth = request.form.get("carwidth")
        carheight = request.form.get("carheight")
        curbweight = request.form.get("curbweight")
        enginetype_ohc = request.form.get("enginetype_ohc")
        cylinder = request.form.get("cylinder")
        enginesize = request.form.get("enginesize")
        fuelsystem = request.form.get("fuelsystem")
        boreratio = request.form.get("boreratio")
        stroke = request.form.get("stroke")
        compression = request.form.get("compression")
        horsepower = request.form.get("horsepower")
        peakrpm = request.form.get("peakrpm")
        citympg = request.form.get("citympg")
        highwaympg = request.form.get("highwaympg")

        # symboling = x_scaler.transform(np.array([symboling]).reshape(1,-1))
        # wheelbase = x_scaler.transform(np.array([wheelbase]).reshape(1,-1))
        # carlength = x_scaler.transform(np.array([carlength]).reshape(1,-1))
        # carwidth = x_scaler.transform(np.array([carwidth]).reshape(1,-1))
        # carheight = x_scaler.transform(np.array([carheight]).reshape(1,-1))
        # curbweight = x_scaler.transform(np.array([curbweight]).reshape(1,-1))
        # enginesize = x_scaler.transform(np.array([enginesize]).reshape(1,-1))
        # boreratio = x_scaler.transform(np.array([boreratio]).reshape(1,-1))
        # stroke = x_scaler.transform(np.array([stroke]).reshape(1,-1))
        # compression = x_scaler.transform(np.array([compression]).reshape(1,-1))
        # horsepower = x_scaler.transform(np.array([horsepower]).reshape(1,-1))
        # peakrpm = x_scaler.transform(np.array([peakrpm]).reshape(1,-1))
        # citympg = x_scaler.transform(np.array([citympg]).reshape(1,-1))
        # highwaympg = x_scaler.transform(np.array([highwaympg]).reshape(1,-1))
        symboling = x_scaler.transform(np.array(symboling).reshape(1,-1))
        wheelbase = x_scaler.transform(np.array([wheelbase]).reshape(1,-1))
        carlength = x_scaler.transform(np.array([carlength]).reshape(1,-1))
        carwidth = x_scaler.transform(np.array([carwidth]).reshape(1,-1))
        carheight = x_scaler.transform(np.array([carheight]).reshape(1,-1))
        curbweight = x_scaler.transform(np.array([curbweight]).reshape(1,-1))
        enginesize = x_scaler.transform(np.array([enginesize]).reshape(1,-1))
        boreratio = x_scaler.transform(np.array([boreratio]).reshape(1,-1))
        stroke = x_scaler.transform(np.array([stroke]).reshape(1,-1))
        compression = x_scaler.transform(np.array([compression]).reshape(1,-1))
        horsepower = x_scaler.transform(np.array([horsepower]).reshape(1,-1))
        peakrpm = x_scaler.transform(np.array([peakrpm]).reshape(1,-1))
        citympg = x_scaler.transform(np.array([citympg]).reshape(1,-1))
        highwaympg = x_scaler.transform(np.array([highwaympg]).reshape(1,-1))

        data = np.append(symboling, [[fueltype, aspiration, door, enginelocation]], axis=1)
        data = np.append(data, wheelbase, axis=1)
        data = np.append(data, carlength, axis=1)
        data = np.append(data, carwidth, axis=1)
        data = np.append(data, carheight, axis=1)
        data = np.append(data, curbweight, axis=1)
        data = np.append(data, enginesize, axis=1)
        data = np.append(data, boreratio, axis=1)
        data = np.append(data, stroke, axis=1)
        data = np.append(data, compression, axis=1)
        data = np.append(data, horsepower, axis=1)
        data = np.append(data, peakrpm, axis=1)
        data = np.append(data, citympg, axis=1)
        data = np.append(data, highwaympg, axis=1)
        data = np.append(data, [[enginetype_ohc, cylinder, drivewheel_fwd, fuelsystem, car_body_sedan]], axis=1)

        # data = np.array([symboling, fueltype, aspiration, door, enginelocation, wheelbase, carlength, carwidth, carheight, curbweight, enginesize, boreratio, stroke, compression, horsepower, peakrpm, citympg, highwaympg, enginetype_ohc, cylinder, drivewheel_fwd, fuelsystem, car_body_sedan]).reshape(1, -1)

        # Sequence in training set
        # ['symboling', 'fueltype', 'aspiration', 'doornumber', 'enginelocation',
        #  'wheelbase', 'carlength', 'carwidth', 'carheight', 'curbweight',
        #  'enginesize', 'boreratio', 'stroke', 'compressionratio', 'horsepower',
        #  'peakrpm', 'citympg', 'highwaympg', 'OHC_Code', 'four_cylinder',
        #  'drivewheel_fwd', 'fuelsystem_mpfi', 'carbody_sedan']

        # scaled_variables = ['symboling', 'wheelbase', 'carlength', 'carwidth', 'carheight', 'curbweight', 'enginesize',
        #                'boreratio', 'stroke', 'compressionratio', 'horsepower', 'peakrpm', 'citympg', 'highwaympg']
        data = np.asarray(data, dtype='float64')
        print('Before ',data)
        print('\n\n')
        my_prediction = model.predict(data)
        print('After ',my_prediction)
        predicted_price = y_scaler.inverse_transform(my_prediction)
        print('Final converted price', predicted_price)
    return render_template('Car_Prediction.html', prediction=predicted_price)

if __name__ == '__main__':
    app.run(debug=True)