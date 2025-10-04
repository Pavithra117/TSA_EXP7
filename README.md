# Ex.No: 07                                       AUTO REGRESSIVE MODEL
### Date: 04/10/2025



### AIM:
To Implementat an Auto Regressive Model using Python
### ALGORITHM:
1. Import necessary libraries
2. Read the CSV file into a DataFrame
3. Perform Augmented Dickey-Fuller test
4. Split the data into training and testing sets.Fit an AutoRegressive (AR) model with 13 lags
5. Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF)
6. Make predictions using the AR model.Compare the predictions with the test data
7. Calculate Mean Squared Error (MSE).Plot the test data and predictions.
### PROGRAM:
Import necessary libraries:
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg
```
Read the CSV file into a DataFrame :
```
data = pd.read_csv('GoogleStockPrices.csv', parse_dates=['Date'], index_col='Date')
```
Perform Augmented Dickey-Fuller test :
```
result = adfuller(data['Close'])
print('ADF Statistic:', result[0])
print('p-value:', result[1])
```
Split the data into training and testing sets :
```
x = int(0.8 * len(data))
train_data = data.iloc[:x]
test_data = data.iloc[x:]
```
Fit an AutoRegressive (AR) model:
```
lag_order = 13
model = AutoReg(train_data['Close'], lags=lag_order)
model_fit = model.fit()
```
Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF) :
```
plt.figure(figsize=(10, 6))
plot_acf(data['Close'], lags=40, alpha=0.05)
plt.title('Autocorrelation Function (ACF) - Close Price')
plt.show()

plt.figure(figsize=(10, 6))
plot_pacf(data['Close'], lags=40, alpha=0.05)
plt.title('Partial Autocorrelation Function (PACF) - Close Price')
plt.show()
```
Make predictions using the AR model :
```
predictions = model_fit.predict(start=len(train_data), end=len(train_data)+len(test_data)-1)
```
Compare the predictions with the test data :
```
mse = np.mean((test_data['Close'].values - predictions.values) ** 2)
print('Mean Squared Error (MSE):', mse)
```
Plot the test data and predictions :
```
plt.figure(figsize=(12, 6))
plt.plot(test_data['Close'], label='Test Data - Close Price')
plt.plot(predictions, label='Predictions - Close Price', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('AR Model Predictions vs Test Data')
plt.legend()
plt.grid()
plt.show()
```

### OUTPUT:

ADF test result:


ADF Statistic: 0.6525354825000549

p-value: 0.9888414405204398
<Figure size 1000x600 with 0 Axes>
  
PACF plot:

<img width="718" height="531" alt="image" src="https://github.com/user-attachments/assets/cd2478bd-15f4-42b3-ad27-ab1944344c19" />

ACF plot:

<img width="730" height="543" alt="image" src="https://github.com/user-attachments/assets/cb78b1cd-4755-4576-86bc-77b784473d3d" />

Accuracy:

Mean Squared Error (MSE): 3378.447475574205

Prediction vs test data:

<img width="943" height="508" alt="image" src="https://github.com/user-attachments/assets/e4f67d7c-4815-4b2d-9f7e-c978211c3d05" />


### RESULT:
Thus we have successfully implemented the auto regression function using python.
