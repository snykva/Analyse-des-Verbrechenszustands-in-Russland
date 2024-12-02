import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

data = pd.read_csv('crime.csv')
date = pd.read_csv('date_crime.txt', names=['date'])

data['month'] = pd.to_datetime(data['month'])
x = pd.to_numeric(data['month']).values.reshape(-1, 1)

date['date'] = pd.to_datetime(date['date'])
x_pred = pd.to_numeric(date['date']).values.reshape(-1, 1)

poly = PolynomialFeatures(degree = 7)
x_trans = poly.fit_transform(x)
x_trans = pd.DataFrame(x_trans)

cor_ar = data.corr(numeric_only=True)
plt.figure(figsize = (16,9))
gm = sns.heatmap(cor_ar, annot = True, cmap = "coolwarm", vmin = -1, vmax = 1)
plt.show()

x_train, x_test, y_train, y_test = train_test_split(x_trans, data[['Total_crimes']], test_size=0.2) #регрессия между датой и общим кол-вом преступлений

sqr1 = LinearRegression()
sqr1.fit(x_train, y_train)
y_pred = sqr1.predict(x_test)

x_plot_tot = np.linspace(x_test[[1]].min(), x_test[[1]].max(), 100)
y_plot_tot = sqr1.predict(poly.fit_transform(x_plot_tot))

y_pred_tot = sqr1.predict(poly.fit_transform(x_pred))

det_tot = r2_score(y_test, y_pred)

plt.figure(figsize=(16, 9))
plt.scatter(x, data[['Total_crimes']], label='общее количество преступлений за месяц')
plt.plot(x_plot_tot, y_plot_tot, color='red', label=f'график полиномиальной регрессии, r2 - {det_tot}')
plt.plot(x_pred, y_pred_tot, 'r--', label='предсказание')
plt.xlabel('даты')
plt.ylabel('количество преступлений')
plt.title('все преступлления')
plt.legend()
plt.show()

x_train, x_test, y_train, y_test = train_test_split(x_trans, data[['Serious']], test_size=0.2)

sqr2 = LinearRegression()
sqr2.fit(x_train, y_train)
y_pred = sqr2.predict(x_test)

x_plot_ser = np.linspace(x_test[[1]].min(), x_test[[1]].max(), 100)
y_plot_ser = sqr2.predict(poly.fit_transform(x_plot_ser))

y_pred_ser = sqr2.predict(poly.fit_transform(x_pred))

det_ser = r2_score(y_test, y_pred)

plt.figure(figsize=(16, 9))
plt.scatter(x, data[['Serious']], label='общее количество тяжких и особо тяжких преступление за месяц')
plt.plot(x_plot_ser, y_plot_ser, color='red', label=f'график полиномиальной регрессии, r2 - {det_ser}')
plt.plot(x_pred, y_pred_ser, 'r--', label='предсказание')
plt.xlabel('даты')
plt.ylabel('количество тяжких и особо тяжких преступление')
plt.title('количетсво тяжких и особо тяжких преступлений за месяц')
plt.legend()
plt.show()

x_train, x_test, y_train, y_test = train_test_split(x_trans, data[['Murder']], test_size=0.2)

sqr3 = LinearRegression()
sqr3.fit(x_train, y_train)
y_pred = sqr3.predict(x_test)

x_plot_mur = np.linspace(x_test[[1]].min(), x_test[[1]].max(), 100)
y_plot_mur = sqr3.predict(poly.fit_transform(x_plot_mur))

y_pred_mur = sqr3.predict(poly.fit_transform(x_pred))

det_mur = r2_score(y_test, y_pred)

plt.figure(figsize=(16, 9))
plt.scatter(x, data[['Murder']], label='общее количество убийств и покушений на убийство за месяц')
plt.plot(x_plot_mur, y_plot_mur, color='red', label=f'график полиномиальной регрессии, r2 - {det_mur}')
plt.plot(x_pred, y_pred_mur, 'r--', label='предсказание')
plt.xlabel('даты')
plt.ylabel('количество убийств и покушений на убийство')
plt.title('убийства и покушения на убийство за месяц')
plt.legend()
plt.show()

x_train, x_test, y_train, y_test = train_test_split(x_trans, data[['Harm_to_health']], test_size=0.2)

sqr8 = LinearRegression()
sqr8.fit(x_train, y_train)
y_pred = sqr8.predict(x_test)

x_plot_har = np.linspace(x_test[[1]].min(), x_test[[1]].max(), 100)
y_plot_har = sqr8.predict(poly.fit_transform(x_plot_har))

y_pred_har = sqr8.predict(poly.fit_transform(x_pred))

det_har = r2_score(y_test, y_pred)

plt.figure(figsize=(16, 9))
plt.scatter(x, data[['Harm_to_health']], label='общее количество преступлений с тяжким нанесением ущерба здоровью за месяц')
plt.plot(x_plot_har, y_plot_har, color='red', label=f'график полиномиальной регрессии, r2 - {det_har}')
plt.plot(x_pred, y_pred_har, 'r--', label='предсказание')
plt.xlabel('даты')
plt.ylabel('количество преступлений с нанесением тяжкого ущерба здоровью')
plt.title('преступления с тяжким нанесением ущерба здоровья за месяц')
plt.legend()
plt.show()