#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 22:24:16 2020

@author: matias
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

################## ENTENDIMIENTO DE LA DATA ##############################
print("informacion del dataset")
print(boston.keys())
print()

########PREPARACION DE LA DATA REGRESION LINEAL SIMPLE################

#seleccionamos columna 5,6 y 7del dataset
X_multiple = boston.data[:,5:8]
print(X_multiple)

#defino datos correspondientes de la etiqueta
Y_multiple= boston.target

###############IMPLEMENTACION DE REGRESION LINEAL#################

from sklearn.model_selection import train_test_split

#separo datos de train en entrenamiento y prueba para probar los algoritmos
X_train, X_test, y_train, y_test = train_test_split(X_multiple,Y_multiple,test_size=0.2)

#defino el algoritmo a utilizar
lr = linear_model.LinearRegression()

#entreno el modelo
lr.fit(X_train, y_train)

#realizo una prediccion
y_pred_multiple = lr.predict(X_test)

#muestro la ecuacion

print("Y = [",lr.coef_,']X + ', lr.intercept_)

print("precision del modelo")
print(lr.score(X_train, y_train))







