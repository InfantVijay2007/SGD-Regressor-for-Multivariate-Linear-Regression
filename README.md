# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the data containing house area, number of rooms, price, and occupants.

2.Scale the input features and create a SGDRegressormodel.

3.Train the model using fit()the given house data.

4.Give a new house's details and use predict()to find its price and number of occupants.

## Program:
```py
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: INFANT VIJAY A
RegisterNumber:  212225240052
*/

import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

# Input data
# [Area in sq.ft, Number of rooms]
X = np.array([
    [1000, 2],
    [1500, 3],
    [2000, 4],
    [2500, 5],
    [3000, 6]
])

# Target values
house_price = np.array([2000000, 3000000, 4000000, 5000000, 6000000])
occupants = np.array([2, 3, 4, 5, 6])

# Scale input data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# SGD Regressor for house price
price_model = SGDRegressor(max_iter=1000, learning_rate='constant',
                           eta0=0.01, random_state=42)
price_model.fit(X_scaled, house_price)

# SGD Regressor for number of occupants
occupant_model = SGDRegressor(max_iter=1000, learning_rate='constant',
                              eta0=0.01, random_state=42)
occupant_model.fit(X_scaled, occupants)

# New house details
new_house = np.array([[2200, 4]])
new_house_scaled = scaler.transform(new_house)

# Predictions
predicted_price = price_model.predict(new_house_scaled)
predicted_occupants = occupant_model.predict(new_house_scaled)

print("Predicted House Price: ₹", round(predicted_price[0], 2))
print("Predicted Number of Occupants:", round(predicted_occupants[0]))

```

## Output:
<img width="989" height="646" alt="image" src="https://github.com/user-attachments/assets/ee994c49-b07d-4d89-9cd8-2056a62cccb0" />


## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
