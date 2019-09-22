# simple linear regresion
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#importing the dataset
dataset=pd.read_csv("Salary_data.csv")
X = dataset.iloc[:, :-1].values
y=dataset.iloc[:, 1].values

#spliting the data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/3,random_state=0)

#feature scaling
'''from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)'''

#fiting the linear model to the train dataset
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

#predict the test data set result
y_predict=regressor.predict(X_test)

#visulaize the traning dataset 
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Salary vs experience(training set)')
plt.xlabel('experience')
plt.ylabel('salary')
plt.show()

#visualizing the test dataset
plt.scatter(X_test,y_test,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Salary vs experience(test set)')
plt.xlabel('experience')
plt.ylabel('salary')
plt.show()
