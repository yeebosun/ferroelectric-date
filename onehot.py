import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_excel('data.exal')
ohe = OneHotEncoder(handle_unknown='ignore')
ohe_column = pd.DataFrame(ohe.fit_transform(data[['crystal system']]).toarray())
data2 = data.drop(['crystal system'], axis=1)
sns.heatmap(data2.corr(), annot=True, cmap='RdBu', xticklabels=1, yticklabels=1)
plt.show()

data2.join(ohe_column)
data2 = pd.DataFrame(data2)

data2.to_excel('onehot.xlsx', index=False)
