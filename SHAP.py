# 导入需要的库
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import numpy as np
import shap
import seaborn as sns



X = pd.DataFrame(pd.read_excel('X_train.xlsx')).values  # 输入特征
Y = pd.DataFrame(pd.read_excel('Y_train.xlsx')).values  # 目标变量

# 测试


model = xgb.XGBRegressor()
model.fit(X, Y)
# xgb.plot_tree(model)
# 使用测试数据预测类别
train_predict = model.predict(X)
# test_predict = model.predict(X_test)
# 线性拟合
lm = LinearRegression()
lm.fit(Y, train_predict)
# shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)
shap_values2 = explainer(X)
# shap.summary_plot(shap_values, X, plot_type="bar")
shap.summary_plot(shap_values, X)
shap.plots.bar(shap_values2)
sns.heatmap(df.corr(), annot=True, cmap='RdBu', xticklabels=1, yticklabels=1)
plt.show()


msetrain = mean_squared_error(Y, train_predict)
R2 = lm.score(Y, train_predict)


