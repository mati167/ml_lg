# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 12:11:31 2020

@author: Matias Gonzalez
"""


#import of libraries
import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt

####PREPARACION DE LA DATA######

#importamos la data de scikit-learn
boston = datasets.load_boston()
print(boston)
print()


########PREPARACION DE LA DATA REGRESION POLINOMIAL################

#seleccionamos solamente la columna 6
X_p = boston.data[:, np.newaxis, 5]

#defino los datos correspondientes a las etiquetas
y_p = boston.target

#graficamos los datos correspondientes
plt.scatter(X_p, y_p)
plt.show()

from sklearn.model_selection import train_test_split
#separo datos train y test
x_train_p,x_test_p, y_train_p,y_test_p = train_test_split(X_p, y_p, test_size=0.2)

from sklearn.preprocessing import PolynomialFeatures

#se define el grado del polinomio
poli_reg = PolynomialFeatures(degree=2)


#se transforma las caracteristicas existentes en polinomio de mayor grado
X_train_poli = poli_reg.fit_transform(x_train_p)
X_test_poli = poli_reg.fit_transform(x_test_p)

#defino el algoritmo a utilizar
pr = linear_model.LinearRegression()

#entreno el modelo
pr.fit(X_train_poli,y_train_p)

#realizo la prediccion
y_pred_pr = pr.predict(X_test_poli)

#grafico los datos junto con el modelo
plt.scatter(x_test_p,y_test_p)
plt.plot(x_test_p,y_pred_pr, color="red",linewidth=3)
plt.show()











 