import joblib
from flask import Flask, render_template, request
import requests
import pickle
from flask_cors import CORS, cross_origin
import numpy as np
import pandas as pd
from datetime import datetime
import crops
import random


app = Flask(__name__, template_folder='templates')

model = joblib.load('crop_app')
model1 = pickle.load(open('classifier.pkl', 'rb'))
ferti = pickle.load(open('fertilizer.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index1.html')

@app.route('/crop', methods=["GET", "POST"])
def recomendation():
    if request.method == 'POST':
        try:
            N = float(request.form['Nitrogen'])
            P = float(request.form['Phosphorus'])
            K = float(request.form['Potassium'])
            Temp = float(request.form['Temperature'])
            Humi = float(request.form['Humidity'])
            Ph = float(request.form['ph'])
            Rain = float(request.form['Rainfall'])

            value = [N ,P ,K , Temp, Humi, Ph, Rain]

            if 0 < Ph <= 14 and Temp < 100 and Humi > 0:
                try:
                    arr = [value]
                    prediction = model.predict(arr)[0]
                    return render_template('crop_result.html', prediction=prediction)
                except FileNotFoundError:
                    return "Error: Crop prediction model 'crop_app' not found."
                except Exception as e:
                    return f"Unexpected error: {str(e)}"
            else:
                return "Sorry... Error in entered values in the form. Please check the entered values and try again."
        except KeyError as e:
            return f"Form key error: {str(e)}"
    return render_template('crop.html')

@app.route('/fertili', methods=['GET','POST'])
def predict1():
    if request.method == 'POST':
        try:
            temp1 = request.form.get('temp')
            humi1 = request.form.get('humi')
            mois1 = request.form.get('mois')
            soil1 = request.form.get('soil')
            crop1 = request.form.get('crop')
            nitro1 = request.form.get('nitro')
            pota1 = request.form.get('pota')
            phosp1 = request.form.get('phosp')

            if None in (temp1, humi1, mois1, soil1, crop1, nitro1, pota1, phosp1):
                return "Error: All form fields must be filled.Please Try Again"
            temp1 = int(temp1)
            humi1 = int(humi1)
            mois1 = int(mois1)
            nitro1 = int(nitro1)
            pota1 = int(pota1)
            phosp1 = int(phosp1)
            if not all(0 <= value <= 100 for value in [temp1, humi1, mois1, nitro1, pota1, phosp1]) or temp1 >= 100:
                return "Error: Input values out of range. Temperature should be 0-99, and other values should be 0-100."

            input_values = [temp1, humi1, mois1, soil1, crop1,
                            nitro1, pota1, phosp1]

            res = ferti.classes_[model1.predict([input_values])]

            return render_template('ferti_result.html', prediction=f'Predicted Fertilizer is {res[0]}')
        except ValueError as e:
            return f"Value error: {str(e)}"
        except Exception as e:
            return f"Unexpected error: {str(e)}"
    return render_template('ferti.html')


@app.route('/weather', methods=['GET', 'POST'])
def weather():
    if request.method == 'POST':
        city1= request.form.get('city')
        api_key = 'cbf1b9f6f3127d65b15aa57c0cd3d28a'
        cw_r = requests.get(f'https://api.openweathermap.org/data/2.5/weather?q={city1}&appid={api_key}')
        if cw_r.status_code != 200:
            return render_template('error.html')
        current_weather_json = cw_r.json()
        forecast_response = requests.get(f'https://api.openweathermap.org/data/2.5/forecast?q={city1}&appid={api_key}')
        forecast_json = forecast_response.json()
        temperature1 = int(current_weather_json['main']['temp'] - 273.15)
        humidity1 = int(current_weather_json['main']['humidity'])
        pressure1 = int(current_weather_json['main']['pressure'])
        wind1 = int(current_weather_json['wind']['speed'])
        condition1 = current_weather_json['weather'][0]['main']
        desc1 = current_weather_json['weather'][0]['description']
        forecast_data = []
        for forecast in forecast_json['list']:
            fdate = forecast['dt_txt'].split()[0]
            ftemperature = int(forecast['main']['temp'] - 273.15)
            fcondition = forecast['weather'][0]['main']
            fdesc = forecast['weather'][0]['description']
            forecast_data.append({'date': fdate, 'temperature': ftemperature, 'condition': fcondition, 'desc': fdesc})
        return render_template('weather.html', temperature1=temperature1, pressure1=pressure1,
                               humidity1=humidity1, city1=city1, condition1=condition1,
                               wind1=wind1, desc1=desc1, forecast_data=forecast_data)
    else:
        return render_template('weather.html')

CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

cors = CORS(app, resources={r"/ticker": {"origins": "http://localhost:port"}})

commodity_dict = {
    "arhar": "static/Arhar.csv", "bajra": "static/Bajra.csv","barley": "static/Barley.csv",
    "copra": "static/Copra.csv","cotton": "static/Cotton.csv","sesamum": "static/Sesamum.csv",
    "gram": "static/Gram.csv","groundnut": "static/Groundnut.csv","jowar": "static/Jowar.csv",
    "maize": "static/Maize.csv","masoor": "static/Masoor.csv","moong": "static/Moong.csv",
    "niger": "static/Niger.csv","paddy": "static/Paddy.csv","ragi": "static/Ragi.csv",
    "rape": "static/Rape.csv","jute": "static/Jute.csv","safflower": "static/Safflower.csv",
    "soyabean": "static/Soyabean.csv","sugarcane": "static/Sugarcane.csv","sunflower": "static/Sunflower.csv",
    "urad": "static/Urad.csv","wheat": "static/Wheat.csv"
}

annual_rainfall = [29, 21, 37.5, 30.7, 52.6, 150, 299, 251.7, 179.2, 70.5, 39.8, 10.9]
base = {
    "Paddy": 1245.5,  "Arhar": 3200,"Bajra": 1175, "Barley": 980, "Copra": 5100,"Cotton": 3600, "Sesamum": 4200,"Gram": 2800,
    "Groundnut": 3700,"Jowar": 1520,"Maize": 1175,"Masoor": 2800,"Moong": 3500,"Niger": 3500,"Ragi": 1500,"Rape":2500,
    "Jute": 1675, "Safflower": 2500,"Soyabean": 2200,"Sugarcane": 2250,"Sunflower": 3700,"Urad": 4300,"Wheat": 1350
}
commodity_list = []


class Commodity:

    def __init__(self, csv_name):
        self.name = csv_name
        dataset = pd.read_csv(csv_name)
        self.X = dataset.iloc[:, :-1].values
        self.Y = dataset.iloc[:, 3].values
        from sklearn.tree import DecisionTreeRegressor
        depth = random.randrange(7,18)
        self.regressor = DecisionTreeRegressor(max_depth=depth)
        self.regressor.fit(self.X, self.Y)
        
    def getPredictedValue(self, value):
        if value[1]>=2019:
            fsa = np.array(value).reshape(1, 3)
            return self.regressor.predict(fsa)[0]
        else:
            c=self.X[:,0:2]
            x=[]
            for i in c:
                x.append(i.tolist())
            fsa = [value[0], value[1]]
            ind = 0
            for i in range(0,len(x)):
                if x[i]==fsa:
                    ind=i
                    break;
            return self.Y[i]

    def getCropName(self):
        a = self.name.split('.')
        return a[0]


@app.route('/cropprice',methods=['GET','Post'])
def index():
    context = {
        "sixmonths": SixMonthsForecast()
    }
    return render_template('crop_price.html', context=context)


@app.route('/commodity/<name>')
def crop_profile(name):
    maxcrop, mincrop, forecastvalues = TwelveMonthsForecast(name)
    prevvalues = TwelveMonthPrevious(name)
    forecast_x = [i[0] for i in forecastvalues]
    forecast_y = [i[1] for i in forecastvalues]
    previous_x = [i[0] for i in prevvalues]
    previous_y = [i[1] for i in prevvalues]
    current_price = CurrentMonth(name)
    crop_data = crops.crop(name)
    if len(crop_data) < 4:
        crop_data.extend(['', '', '', ''])  
    
    context = {
        "name": name,
        "max_crop": maxcrop,
        "min_crop": mincrop,
        "forecast_values": forecastvalues,
        "forecast_x": str(forecast_x),
        "forecast_y": forecast_y,
        "previous_values": prevvalues,
        "previous_x": previous_x,
        "previous_y": previous_y,
        "current_price": current_price,
        "image_url": crop_data[0],
        "prime_loc": crop_data[1],
        "type_c": crop_data[2],
        "export": crop_data[3]
    }
    return render_template('crop_price_result.html', context=context)

@app.route('/ticker/<item>/<number>')
@cross_origin(origin='localhost',headers=['Content- Type','Authorization'])
def ticker(item, number):
    n = int(number)
    i = int(item)
    data = SixMonthsForecast()
    context = str(data[n][i])

    if i == 2 or i == 5:
        context = '₹' + context
    elif i == 3 or i == 6:

        context = context + '%'
    return context;

def SixMonthsForecast():
    month1=[]
    month2=[]
    month3=[]
    month4=[]
    month5=[]
    month6=[]
    for i in commodity_list:
        crop=SixMonthsForecastHelper(i.getCropName())
        k=0
        for j in crop:
            time = j[0]
            price = j[1]
            change = j[2]
            if k==0:
                month1.append((price,change,i.getCropName().split("/")[1],time))
            elif k==1:
                month2.append((price,change,i.getCropName().split("/")[1],time))
            elif k==2:
                month3.append((price,change,i.getCropName().split("/")[1],time))
            elif k==3:
                month4.append((price,change,i.getCropName().split("/")[1],time))
            elif k==4:
                month5.append((price,change,i.getCropName().split("/")[1],time))
            elif k==5:
                month6.append((price,change,i.getCropName().split("/")[1],time))
            k+=1
    month1.sort()
    month2.sort()
    month3.sort()
    month4.sort()
    month5.sort()
    month6.sort()
    crop_month=[]
    crop_month.append([month1[0][3],month1[len(month1)-1][2],month1[len(month1)-1][0],
                       month1[len(month1)-1][1],month1[0][2],month1[0][0],month1[0][1]])
    crop_month.append([month2[0][3],month2[len(month2)-1][2],month2[len(month2)-1][0],
                       month2[len(month2)-1][1],month2[0][2],month2[0][0],month2[0][1]])
    crop_month.append([month3[0][3],month3[len(month3)-1][2],month3[len(month3)-1][0],
                       month3[len(month3)-1][1],month3[0][2],month3[0][0],month3[0][1]])
    crop_month.append([month4[0][3],month4[len(month4)-1][2],month4[len(month4)-1][0],
                       month4[len(month4)-1][1],month4[0][2],month4[0][0],month4[0][1]])
    crop_month.append([month5[0][3],month5[len(month5)-1][2],month5[len(month5)-1][0],
                       month5[len(month5)-1][1],month5[0][2],month5[0][0],month5[0][1]])
    crop_month.append([month6[0][3],month6[len(month6)-1][2],month6[len(month6)-1][0]
                       ,month6[len(month6)-1][1],month6[0][2],month6[0][0],month6[0][1]])

    return crop_month

def SixMonthsForecastHelper(name):
    cmonth = datetime.now().month
    cyear = datetime.now().year
    crainfall = annual_rainfall[cmonth - 1]
    name = name.split("/")[1]
    name = name.lower()
    commodity = commodity_list[0]
    for i in commodity_list:
        if name == str(i):
            commodity = i
            break
    month_with_year = []
    for i in range(1, 7):
        if cmonth + i <= 12:
            month_with_year.append((cmonth + i, cyear, annual_rainfall[cmonth + i - 1]))
        else:
            month_with_year.append((cmonth + i - 12, cyear + 1, annual_rainfall[cmonth + i - 13]))
    wpis = []
    current_wpi = commodity.getPredictedValue([float(cmonth), cyear, crainfall])
    change = []

    for m, y, r in month_with_year:
        current_predict = commodity.getPredictedValue([float(m), y, r])
        wpis.append(current_predict)
        change.append(((current_predict - current_wpi) * 100) / current_wpi)

    crop_price = []
    for i in range(0, len(wpis)):
        m, y, r = month_with_year[i]
        x = datetime(y, m, 1)
        x = x.strftime("%b %y")
        crop_price.append([x, round((wpis[i]* base[name.capitalize()]) / 100, 2) , round(change[i], 2)])

    return crop_price

def CurrentMonth(name):
    cmonth = datetime.now().month
    cyear = datetime.now().year
    crainfall = annual_rainfall[cmonth - 1]
    name = name.lower()
    commodity = commodity_list[0]
    for i in commodity_list:
        if name == str(i):
            commodity = i
            break
    cwpi = commodity.getPredictedValue([float(cmonth), cyear, crainfall])
    cprice = (base[name.capitalize()]*cwpi)/100
    return cprice

def TwelveMonthsForecast(name):
    cmonth = datetime.now().month
    cyear = datetime.now().year
    crainfall = annual_rainfall[cmonth - 1]
    name = name.lower()
    commodity = commodity_list[0]
    for i in commodity_list:
        if name == str(i):
            commodity = i
            break
    month_with_year = []
    for i in range(1, 13):
        if cmonth + i <= 12:
            month_with_year.append((cmonth + i, cyear, annual_rainfall[cmonth + i - 1]))
        else:
            month_with_year.append((cmonth + i - 12, cyear + 1, annual_rainfall[cmonth + i - 13]))
    max_index = 0
    min_index = 0
    max_value = 0
    min_value = 9999
    wpis = []
    cwpi = commodity.getPredictedValue([float(cmonth), cyear, crainfall])
    change = []

    for m, y, r in month_with_year:
        cpredict = commodity.getPredictedValue([float(m), y, r])
        if cpredict > max_value:
            max_value = cpredict
            max_index = month_with_year.index((m, y, r))
        if cpredict < min_value:
            min_value = cpredict
            min_index = month_with_year.index((m, y, r))
        wpis.append(cpredict)
        change.append(((cpredict - cwpi) * 100) / cwpi)

    max_month, max_year, r1 = month_with_year[max_index]
    min_month, min_year, r2 = month_with_year[min_index]
    min_value = min_value * base[name.capitalize()] / 100
    max_value = max_value * base[name.capitalize()] / 100
    crop_price = []
    for i in range(0, len(wpis)):
        m, y, r = month_with_year[i]
        x = datetime(y, m, 1)
        x = x.strftime("%b %y")
        crop_price.append([x, round((wpis[i]* base[name.capitalize()]) / 100, 2) , round(change[i], 2)])
    x = datetime(max_year,max_month,1)
    x = x.strftime("%b %y")
    max_crop = [x, round(max_value,2)]
    x = datetime(min_year, min_month, 1)
    x = x.strftime("%b %y")
    min_crop = [x, round(min_value,2)]

    return max_crop, min_crop, crop_price


def TwelveMonthPrevious(name):
    name = name.lower()
    cmonth = datetime.now().month
    cyear = datetime.now().year
    commodity = commodity_list[0]
    wpis = []
    crop_price = []
    for i in commodity_list:
        if name == str(i):
            commodity = i
            break
    month_with_year = []
    for i in range(1, 13):
        if cmonth - i >= 1:
            month_with_year.append((cmonth - i, cyear, annual_rainfall[cmonth - i - 1]))
        else:
            month_with_year.append((cmonth - i + 12, cyear - 1, annual_rainfall[cmonth - i + 11]))

    for m, y, r in month_with_year:
        current_predict = commodity.getPredictedValue([float(m), 2013, r])
        wpis.append(current_predict)

    for i in range(0, len(wpis)):
        m, y, r = month_with_year[i]
        x = datetime(y,m,1)
        x = x.strftime("%b %y")
        crop_price.append([x, round((wpis[i]* base[name.capitalize()]) / 100, 2)])
    new_crop_price =[]
    for i in range(len(crop_price)-1,-1,-1):
        new_crop_price.append(crop_price[i])
    return new_crop_price


if __name__ == "__main__":
    arhar = Commodity(commodity_dict["arhar"])
    commodity_list.append(arhar)
    bajra = Commodity(commodity_dict["bajra"])
    commodity_list.append(bajra)
    barley = Commodity(commodity_dict["barley"])
    commodity_list.append(barley)
    copra = Commodity(commodity_dict["copra"])
    commodity_list.append(copra)
    cotton = Commodity(commodity_dict["cotton"])
    commodity_list.append(cotton)
    sesamum = Commodity(commodity_dict["sesamum"])
    commodity_list.append(sesamum)
    gram = Commodity(commodity_dict["gram"])
    commodity_list.append(gram)
    groundnut = Commodity(commodity_dict["groundnut"])
    commodity_list.append(groundnut)
    jowar = Commodity(commodity_dict["jowar"])
    commodity_list.append(jowar)
    maize = Commodity(commodity_dict["maize"])
    commodity_list.append(maize)
    masoor = Commodity(commodity_dict["masoor"])
    commodity_list.append(masoor)
    moong = Commodity(commodity_dict["moong"])
    commodity_list.append(moong)
    niger = Commodity(commodity_dict["niger"])
    commodity_list.append(niger)
    paddy = Commodity(commodity_dict["paddy"])
    commodity_list.append(paddy)
    ragi = Commodity(commodity_dict["ragi"])
    commodity_list.append(ragi)
    rape = Commodity(commodity_dict["rape"])
    commodity_list.append(rape)
    jute = Commodity(commodity_dict["jute"])
    commodity_list.append(jute)
    safflower = Commodity(commodity_dict["safflower"])
    commodity_list.append(safflower)
    soyabean = Commodity(commodity_dict["soyabean"])
    commodity_list.append(soyabean)
    sugarcane = Commodity(commodity_dict["sugarcane"])
    commodity_list.append(sugarcane)
    sunflower = Commodity(commodity_dict["sunflower"])
    commodity_list.append(sunflower)
    urad = Commodity(commodity_dict["urad"])
    commodity_list.append(urad)
    wheat = Commodity(commodity_dict["wheat"])
    commodity_list.append(wheat)

    app.run()



    # app.run()


# if __name__ == '__main__':
#     app.run(debug=True)
