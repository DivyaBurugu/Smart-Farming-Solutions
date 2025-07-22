import pandas as pd
df = pd.read_csv('Crop_recommendation.csv')
print(df)
x = df.drop('label', axis = 1)
y = df['label']
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, stratify = y, random_state = 1,test_size=0.2)
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
model = GaussianNB()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
naiveacc= accuracy_score(y_test, y_pred)
print("Accuracy of naive_bayes is " + str(naiveacc))
import joblib 
file_name = 'crop_app'
joblib.dump(model,'crop_app')
app = joblib.load('crop_app')




