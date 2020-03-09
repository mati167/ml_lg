# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 13:51:52 2020

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

########PREPARACION DE LA DATA REGRESION LINEAL SIMPLE################

#seleccionamos columna 5 del dataset
X = boston.data[:,np.newaxis, 5]

#defino datos correspondientes de la etiqueta
Y= boston.target

#grafico datos correspondientes
plt.scatter(X, Y)
plt.xlabel("numero de habitacions")
plt.ylabel("valor medio")
plt.show()


###############IMPLEMENTACION DE REGRESION LINEAL#################

from sklearn.model_selection import train_test_split

#separo datos de train en entrenamiento y prueba para probar los algoritmos
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2)

#defino el algoritmo a utilizar
lr = linear_model.LinearRegression()

#entreno el modelo
lr.fit(X_train, y_train)

#realizo una prediccion
y_pred = lr.predict(X_test)

#grafico datos correspondientes con el modelo
plt.scatter(X_test, y_test)
plt.plot(X_test,y_pred,color="red")
plt.title("Regresion Lineal Simple")
plt.xlabel("numero de habitacions")
plt.ylabel("valor medio")
plt.show()


#muestro la ecuacion

print("Y = ",lr.coef_,'X + ', lr.intercept_)

print("precision del modelo")
print(lr.score(X_train, y_train))














