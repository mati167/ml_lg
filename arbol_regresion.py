#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 00:22:34 2020

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



###############PREPARA LA DATA ARBOLES DE DECISION REGRESION ############

X_adr = boston.data[:,np.newaxis,5]

#defino los datos correspondientes a las etiquetas
y_adr = boston.target

#Graficamos los datos correspondientes
plt.scatter(X_adr, y_adr)
plt.show()


#################### IMPLEMENTACION DE ARBOL DE DECISION REGRESION ################

from sklearn.model_selection import train_test_split

#separo los datos en train y test
X_train, X_test, y_train, y_test = train_test_split(X_adr, y_adr,test_size=0.2)

from sklearn.tree import DecisionTreeRegressor

#defino el algoritmo a utilizar
adr = DecisionTreeRegressor(max_depth = 5)

#Entreno el modelo
adr.fit(X_train, y_train)

#realizo una prediccion
y_pred = adr.predict(X_test)

#graficamos los datos de prueba junto con la prediccion
X_grid = np.arange(min(X_test), max(X_test), 0.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X_test, y_test)
plt.plot(X_grid, adr.predict(X_grid),color='red', linewidth=3 )
plt.show()



















