# Import Package
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from joblib import dump, load
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Read Dataset
df = pd.read_csv("../dataset/housePriceTehran.csv")
df = df.drop("Price(USD)",axis=1)
df = df.drop("Address",axis=1)
df["Parking"] = df["Parking"].astype(int)
df["Elevator"] = df["Elevator"].astype(int)
df["Warehouse"] = df["Warehouse"].astype(int)

# Variable value
x = df.drop("Price",axis=1)
y = df["Price"]

# Convert to array
x = np.array(x)
y = np.array(y)

# Normalize data
scaler = StandardScaler()
x = scaler.fit_transform(x)

# Split Train and Test
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=.1)

# Draw Plot
plt.plot(x_train,y_train,"*")

# RandomForestRegressor
model = RandomForestRegressor(n_estimators=110)
model.fit(x_train,y_train)

# Predict
y_pred_train = model.predict(x_train)
y_pred_test = model.predict(x_test)

# Accuracy
score = r2_score(y_train,y_pred_train)
print("The accuracy of our model is {}%".format(round(score, 2) *100))
score_2 = r2_score(y_test,y_pred_test)
print("The accuracy of our model is {}%".format(round(score_2, 2) *100))

# Save And Load Model
dump(model, "../model/randomForsetHouseTehran.joblib.joblib") 
# model = load("../model/randomForsetHouseTehran.joblib.joblib")