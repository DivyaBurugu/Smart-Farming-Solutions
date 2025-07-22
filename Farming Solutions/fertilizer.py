# import pandas as pd
# import warnings
# warnings.filterwarnings('ignore')
# d = pd.read_csv('Fertilizer Prediction.csv')
# d.rename(columns={'Humidity ':'Humidity','Soil Type':'Soil_Type','Crop Type':'Crop_Type','Fertilizer Name':'Fertilizer'},inplace=True)
# d.describe(include='all')
# from sklearn.preprocessing import LabelEncoder
# encode_soil = LabelEncoder()
# d.Soil_Type = encode_soil.fit_transform(d.Soil_Type)
# Soil_Type = pd.DataFrame(zip(encode_soil.classes_,encode_soil.transform(encode_soil.classes_)),columns=['Original','Encoded'])
# Soil_Type = Soil_Type.set_index('Original')
# encode_crop = LabelEncoder()
# d.Crop_Type = encode_crop.fit_transform(d.Crop_Type)
# Crop_Type = pd.DataFrame(zip(encode_crop.classes_,encode_crop.transform(encode_crop.classes_)),columns=['Original','Encoded'])
# Crop_Type = Crop_Type.set_index('Original')
# encode_ferti = LabelEncoder()
# d.Fertilizer = encode_ferti.fit_transform(d.Fertilizer)
# Fertilizer = pd.DataFrame(zip(encode_ferti.classes_,encode_ferti.transform(encode_ferti.classes_)),columns=['Original','Encoded'])
# Fertilizer = Fertilizer.set_index('Original')
# Fertilizer
# from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(d.drop('Fertilizer',axis=1),d.Fertilizer,test_size=0.2,random_state=1)
# from sklearn.metrics import classification_report
# from sklearn.ensemble import RandomForestClassifier
# r = RandomForestClassifier()
# pred_r = r.fit(x_train,y_train).predict(x_test)
# print(classification_report(y_test,pred_r))
# from sklearn.model_selection import GridSearchCV
# params = {
#     'n_estimators':[300,400,500],
#     'max_depth':[5,10,15],
#     'min_samples_split':[2,5,8]
# }
# grid_r = GridSearchCV(r,params,cv=3,verbose=3,n_jobs=-1)
# grid_r.fit(x_train,y_train)
# pred_r = grid_r.predict(x_test)
# params = {
#     'n_estimators':[350,400,450],
#     'max_depth':[2,3,4,5,6,7],
#     'min_samples_split':[2,5,8]
# }
# grid_r = GridSearchCV(r,params,cv=3,verbose=3,n_jobs=-1)
# grid_r.fit(d.drop('Fertilizer',axis=1),d.Fertilizer)
# import pickle
# pickle_out = open('classifier.pkl','wb')
# pickle.dump(grid_r,pickle_out)
# pickle_out.close()
# model2 = pickle.load(open('classifier.pkl','rb'))
# model2.predict([[34,67,62,0,1,7,0,30]])
# # import pickle
# pickle_out = open('fertilizer.pkl','wb')
# pickle.dump(encode_ferti,pickle_out)
# pickle_out.close()
# ferti = pickle.load(open('fertilizer.pkl','rb'))
# ferti.classes_[6]
# from sklearn.metrics import accuracy_score

# # Compute accuracy
# accuracy = accuracy_score(y_test, pred_r)
# print(f'Accuracy: {accuracy:.2f}')

import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import optuna
import pickle

# Load and preprocess data
d = pd.read_csv('Fertilizer Prediction.csv')
d.rename(columns={'Humidity ':'Humidity','Soil Type':'Soil_Type','Crop Type':'Crop_Type','Fertilizer Name':'Fertilizer'}, inplace=True)

encode_soil = LabelEncoder()
d.Soil_Type = encode_soil.fit_transform(d.Soil_Type)

encode_crop = LabelEncoder()
d.Crop_Type = encode_crop.fit_transform(d.Crop_Type)

encode_ferti = LabelEncoder()
d.Fertilizer = encode_ferti.fit_transform(d.Fertilizer)

x_train, x_test, y_train, y_test = train_test_split(
    d.drop('Fertilizer', axis=1),
    d.Fertilizer,
    test_size=0.2,
    random_state=1
)

# Define Optuna objective
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 2, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
    }
    clf = RandomForestClassifier(**params)
    clf.fit(x_train, y_train)
    preds = clf.predict(x_test)
    return accuracy_score(y_test, preds)

# Run Optuna study
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# Train final model using best parameters
best_params = study.best_params
final_model = RandomForestClassifier(**best_params)
final_model.fit(d.drop('Fertilizer', axis=1), d.Fertilizer)

# Save model and label encoder
with open('classifier.pkl', 'wb') as f:
    pickle.dump(final_model, f)

with open('fertilizer.pkl', 'wb') as f:
    pickle.dump(encode_ferti, f)

# Test prediction
model2 = pickle.load(open('classifier.pkl', 'rb'))
print(model2.predict([[34, 67, 62, 0, 1, 7, 0, 30]]))

# Evaluate
y_pred = final_model.predict(x_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred):.2f}')

