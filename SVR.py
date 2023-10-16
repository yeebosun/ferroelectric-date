# 导入需要的库
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import numpy as np
import seaborn as sns


X = pd.DataFrame(pd.read_excel('X_train.xlsx')).values  # 输入特征
Y = pd.DataFrame(pd.read_excel('Y_train.xlsx')).values  # 目标变量
X_v = pd.DataFrame(pd.read_excel('X_validation.xlsx')).values  # 输入特征
Y_v = pd.DataFrame(pd.read_excel('Y_validation.xlsx')).values  # 目标变量
X_t = pd.DataFrame(pd.read_excel('X_test.xlsx')).values  # 输入特征
Y_t = pd.DataFrame(pd.read_excel('Y_test.xlsx')).values  # 目标变量
model = SVR(
    kernel='rbf',
    gamma=0.62,
    tol=0.001,
    C=1.5,
    epsilon=0.2)

model.fit(X, Y)
train_predict = model.predict(X)
train_predict_v = model.predict(X_v)
train_predict_t = model.predict(X_t)
lm = LinearRegression()
lm.fit(Y, train_predict)
MAE = mean_squared_error(Y, train_predict)
R2 = lm.score(Y, train_predict)
print(MAE)
print(R2)

lm.fit(Y_v, train_predict_v)
MAE_v = mean_squared_error(Y_v, train_predict_v)
R2_v = lm.score(Y_v, train_predict_v)
print(MAE_v)
print(R2_v)

lm.fit(Y_t, train_predict_t)
MAE_t = mean_squared_error(Y_t, train_predict_t)
R2_t = lm.score(Y_t, train_predict_t)
print(MAE_t)
print(R2_t)

