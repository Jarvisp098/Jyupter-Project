import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

#Genarate synthetic data
np.random.seed(0)
X = 2 * np.random.rand(100, 1) #genrate 100 random numbers between 0 & 2
y = 4 + 3 * X + np.random.randn(100, 1) #linear relationship with noise

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_prediction = model.predict(X_test)

mse = mean_squared_error(y_test, y_prediction)
plt.scatter(X_test, y_test, color='blue')
plt.plot(X_test, y_prediction, color='red')
plt.xlabel('X')
plt.ylabel('y')
plt.title(mse)
plt.show()

print(f"Mean Squared Error: {mse}")