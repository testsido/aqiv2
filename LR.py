import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv('./Real_Combine.csv')

df=df.dropna()
X=df.iloc[:,:-1]
# X=df.iloc[:,[-9,-8,-7,-6]] ## independent features
y=df.iloc[:,-1]
# from sklearn.ensemble import ExtraTreesRegressor
# import matplotlib.pyplot as plt
# model = ExtraTreesRegressor()
# model.fit(X,y)
from sklearn import metrics
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X,y)
# print("Coefficient of determination R^2 <-- on train set: {}".format(regressor.score(X_train, y_train)))
# print("Coefficient of determination R^2 <-- on train set: {}".format(regressor.score(X_test, y_test)))
# from sklearn.model_selection import cross_val_score
# score=cross_val_score(regressor,X,y,cv=5)
# score.mean()
# prediction=regressor.predict(X_test)
# accuracy_rate = []

# # Will take some time
# for i in range(1,40):
    
#     knn = KNeighborsRegressor(n_neighbors=i)
#     score=cross_val_score(knn,X,y,cv=10,scoring="neg_mean_squared_error")
#     accuracy_rate.append(score.mean())

# FIRST A QUICK COMPARISON TO OUR ORIGINAL K=1
# knn = KNeighborsRegressor(n_neighbors=1)

# knn.fit(X_train,y_train)
# predictions = knn.predict(X_test)

# from sklearn import metrics
# print('MAE:', metrics.mean_absolute_error(y_test, predictions))
# print('MSE:', metrics.mean_squared_error(y_test, predictions))
# print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

# # FIRST A QUICK COMPARISON TO OUR ORIGINAL K=1
# knn = KNeighborsRegressor(n_neighbors=3)

# knn.fit(X_train,y_train)
# predictions = knn.predict(X_test)



# from sklearn import metrics
# print('MAE:', metrics.mean_absolute_error(y_test, predictions))
# print('MSE:', metrics.mean_squared_error(y_test, predictions))
# print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

import pickle
# open a file, where you ant to store the data
file = open('LinearModel.pkl', 'wb')

# dump information to that file
pickle.dump(regressor, file)
# model = pickle.load(open('LinearModel.pkl','rb'))