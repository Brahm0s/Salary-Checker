# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.model_selection import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.2, random_state = 0)"""

# Fiting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
linReg = LinearRegression()
linReg.fit(x, y)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
polyReg = PolynomialFeatures(degree = 3)
xPoly = polyReg.fit_transform(x)

linReg2 = LinearRegression()
linReg2.fit(xPoly, y)

# Visualising the Linear Regression results
plt.scatter(x, y, color = 'red')
plt.plot(x, linReg.predict(x), color = 'blue')
plt.title('Truth or Bluff{Linear Regression}')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Visualising the Plynomial Regression results
xGrid = np.arange(min(x), max(x), 0.1)
xGrid = xGrid.reshape((len(xGrid), 1))

plt.scatter(x, y, color = 'red')
plt.plot(x, linReg2.predict(polyReg.fit_transform(x)), color = 'blue')
plt.title('Truth or Bluff{Linear Regression}')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#Predicting a new result with Linear Regression
linReg.predict([6.5])

#Predicting a new result with Polynomial Regression
linReg2.predict(polyReg.fit_transform([6.5]))