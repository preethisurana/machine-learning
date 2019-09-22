#SupportVectorRegression
#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing the dataset
dataset=pd.read_csv('Position_Salaries.csv')
X=dataset.iloc[: 1:2].values
y=dataset.iloc[: 2].values

#spliting the dataset
'''from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.5,random_state=0)'''

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
sc_y=StandardScaler()
X=sc_X.fit_transform(X)
y=sc_y.fit_transform(y)
 
#fitting svr into the dataset
from sklearn.svm import SVR
regressor=SVR(kernel='rbf')
regressor.fit(X,y) 

#predicting the result(if you want to know what will the salary of a 6.5experience person should be )
y_pred=regressor.predict(sc_X.transform(np.array[[(6.5)]]))
y_pred=sc_y.inverse(y_pred)#inverse to get back the original values 

# Visualising the SVR results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.01) 
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Support Vector Regression')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()