import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_csv("canada_per_capita_income.csv")
# print(df[['per capita income (US$)']])
# %matplotlib.inline
model = linear_model.LinearRegression()
model.fit(df[['year']],df['per capita income (US$)'])
plt.scatter(df.year,df[['per capita income (US$)']])
plt.plot(df[['year']],model.predict(df[['year']]),color='red')
plt.show()
x = model.predict([[2020]])
df.loc[len(df.index)] = [2020,x[0]]
df.to_csv('Output.csv',index=False)
