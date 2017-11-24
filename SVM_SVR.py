
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,[1]].values
y = dataset.iloc[:,[2]].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y= sc_y.fit_transform(y)


from sklearn.svm import SVR
svr=SVR(kernel='rbf')
svr.fit(X,y)
Y_pred=svr.predict(6.5)
Y_pred

plt.scatter(X,y, color='blue')
plt.plot(X, svr.predict(X), color='green')
plt.title('shdjkawhdw')
plt.xlabel('positionsalaries')
plt.ylabel('salary')
plt.show()