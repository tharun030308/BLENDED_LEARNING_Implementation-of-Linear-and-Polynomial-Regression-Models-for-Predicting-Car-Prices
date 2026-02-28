# BLENDED_LEARNING
# Implementation-of-Linear-and-Polynomial-Regression-Models-for-Predicting-Car-Prices

## AIM:
To write a program to predict car prices using Linear Regression and Polynomial Regression models.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import Libraries: Bring in essential libraries such as pandas, numpy, matplotlib, and sklearn.
2. Load Dataset: Import the dataset containing car prices along with relevant features.
3. Data Preprocessing: Manage missing data and select key features for the model, if required.
4. Split Data: Divide the dataset into training and testing subsets.
5. Train Model: Build a linear regression model and train it using the training data.
6. Make Predictions: Apply the model to predict outcomes for the test set.
7. Evaluate Model: Measure the model's performance using metrics like R² score, Mean Absolute Error (MAE), etc.
8. Check Assumptions: Plot residuals to verify assumptions like homoscedasticity, normality, and linearity.
9. Output Results: Present the predictions and evaluation metrics.

## Program:
```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt



#Load Data
df=pd.read_csv('encoded_car_data (1).csv')
print(df.head())

x=df[['enginesize','horsepower','citympg','highwaympg']]
y=df['price']

#Split Data
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2, random_state=42)


#1. Linear Regression (with scaling)
lr=Pipeline([
    ('scaler',StandardScaler()),
    ('model',LinearRegression())
])
lr.fit(x_train,y_train)
y_pred_linear=lr.predict(x_test)

#2. Polynomial Regression  (degree=2)
poly_model=Pipeline([
    ('poly',PolynomialFeatures(degree=2)),
    ('scaler',StandardScaler()),
    ('model',LinearRegression())
])
poly_model.fit(x_train,y_train)
y_pred_poly=poly_model.predict(x_test)


#Evaluate models
print('Name:Barath B')
print('Reg no:25011113')

print('Linear Regression')
print('MSE =',mean_squared_error(y_test,y_pred_linear))
print('MAE =',mean_absolute_error(y_test,y_pred_linear))
print('R2 Score =',r2_score(y_test,y_pred_linear))


print('\nPolynomial Regression')
print('MSE =',mean_squared_error(y_test,y_pred_poly))
print('MAE =',mean_absolute_error(y_test,y_pred_poly))
print('R2 Score =',r2_score(y_test,y_pred_poly))

#Plot actual vs predicted
plt.figure(figsize=(10,5))
plt.scatter(y_test, y_pred_linear, label='Linear', alpha=0.6)
plt.scatter(y_test, y_pred_poly, label='Polynomial (degree=2)',alpha=0.6)
plt.plot([y.min(),y.max()],[y.min(),y.max()],'r--',label='Perfect Prediction')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Linear vs Polynomial Regression")
plt.legend()
plt.show()
```

## Output:
<img width="808" height="713" alt="image" src="https://github.com/user-attachments/assets/3b5dc455-4570-47ad-bcac-db25a4432878" />
<img width="1181" height="587" alt="image" src="https://github.com/user-attachments/assets/e05e1106-c321-4d9b-85bb-cd042fbafeef" />
## Result:
Thus, the program to implement Linear and Polynomial Regression models for predicting car prices was written and verified using Python programming.
