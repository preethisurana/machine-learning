#multiple regression
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#importing the dataset

dataset=pd.read_csv("Startups.csv")
X = dataset.iloc[:, :-1].values
y=dataset.iloc[:, 4].values

#encoding the categorial variable
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X=LabelEncoder()
X[:,3]=labelencoder_X.fit_transform(X[:, 3])
onehotencoder=OneHotEncoder(categorical_features=[3])
X=onehotencoder.fit_transform(X).toarray()

#avoidind dummy variable trap
X=X[:, 1:]
#spliting the data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


#fitiing the linear model to the train dataset
#the model will include the constant value ir b0(y=b0+b1x1+b2x2..)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

#predicting the model
y_pred=regressor.predict(X_test)
#backward elimination method using ordinary least squares
import statsmodels.formula.api as sm
#adding constant coloumn of ones to support the constant in the formula 
X = np.append(arr=np.ones((50,1)).astype(int),values = X,axis =1)

X_optimal = X[:,[0,1,2,3,4,5]]
regressor_ols = sm.OLS(endog = y,exog = X_optimal).fit()
regressor_ols.summary()


X_optimal = X[:,[0,3,4,5]]
regressor_ols = sm.OLS(endog = y,exog = X_optimal).fit()
regressor_ols.summary()


X_optimal = X[:,[0,3]]
regressor_ols = sm.OLS(endog = y,exog = X_optimal).fit()
regressor_ols.summary()
