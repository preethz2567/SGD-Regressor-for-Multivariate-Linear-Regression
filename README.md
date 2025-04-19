# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. **Import necessary libraries**: Utilize `numpy` for numerical computations, `pandas` for data manipulation, and essential modules from `sklearn` for machine learning operations.  
2. **Load the dataset**: Retrieve the California Housing dataset using `fetch_california_housing()` and convert it into a pandas DataFrame for easier data handling.  
3. **Prepare the feature and target variables**: Define the feature set `x` by dropping the `AveOccup` and `HousingPrice` columns from the DataFrame. Set the target variable `y` to include both `AveOccup` and `HousingPrice`.  
4. **Split the dataset**: Divide the data into training and testing sets using `train_test_split()`, allocating 20% of the data to the test set for evaluation.  
5. **Initialize data scalers**: Use `StandardScaler` to standardize both feature and target variables.  
6. **Scale the data**: Fit the scalers on the training data, then transform both the training and test sets for features (`x_train`, `x_test`) and targets (`y_train`, `y_test`).  
7. **Configure the model**: Set parameters like the maximum number of iterations and stopping tolerance for `SGDRegressor` to control the training process.  
8. **Enable multi-target regression**: Wrap the `SGDRegressor` within `MultiOutputRegressor` to handle prediction of multiple output variables.  
9. **Train the model**: Fit the multi-output regression model using the scaled training data.  
10. **Make predictions**: Use the trained model to predict the target variables for the scaled test data.  
11. **Reverse scaling**: Apply inverse transformation to both the predicted and actual test target values to restore them to their original scale.  
12. **Evaluate performance**: Compute the Mean Squared Error (MSE) to measure the model's accuracy in predicting target values.  
13. **Display results**: Output the first five predicted values to observe the model’s performance.


## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: PREETHI D
RegisterNumber:  212224040250
*/
```
``` 

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

dataset=fetch_california_housing()
df=pd.DataFrame(dataset.data,columns=dataset.feature_names)
df['HousingPrice']=dataset.target
print(df.head())
```
![image](https://github.com/user-attachments/assets/6ba7c8f3-d491-470e-847a-bbfae50d06db)

``` 

X=df.drop(columns=['AveOccup','HousingPrice'])
Y=df[['AveOccup','HousingPrice']]

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

scaler_X=StandardScaler()
scaler_Y=StandardScaler()

X_train = scaler_X.fit_transform(X_train) 
X_test = scaler_X.transform(X_test) 
Y_train = scaler_Y.fit_transform(Y_train) 
Y_test = scaler_Y.transform(Y_test)

```
```
sgd =  SGDRegressor(max_iter=1000, tol=1e-3) 

multi_output_sgd= MultiOutputRegressor(sgd) 
multi_output_sgd.fit(X_train, Y_train) 

Y_pred=multi_output_sgd.predict(X_test) 

Y_test= scaler_Y.inverse_transform(Y_pred)  
 
mse= mean_squared_error (Y_test, Y_pred) 
print("Mean Squared Error:", mse) 

print("\nPredictions: \n",Y_pred[:5])

```
![image](https://github.com/user-attachments/assets/f8488e5d-ec4e-46de-89af-546ad217bb2b)

## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
