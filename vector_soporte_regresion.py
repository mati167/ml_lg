# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
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



###############PREPARA LA DATA VECTORES DE SOPORTE REGRESION ############

#SELECCIONAMOS SOLAMENTE COLUMNA 6
X_svr = boston.data[:,np.newaxis, 5]

#defino los datos correspondientes a las etiquetas
y_svr = boston.target


#graficamos los datos
plt.scatter(X_svr, y_svr)
plt.show()

########### IMPLEMENTACION DE VECTORES DE SOPORTE REGRESION ######

from sklearn.model_selection import train_test_split

#separo los datos train y test
X_train, X_test, y_train, y_test = train_test_split(X_svr,y_svr,test_size=0.2)

from sklearn.svm import SVR

#defino el algoritmo a utilizar
#svr = SVR(kernel='linear', C=1.0, epsilon=0.2)
svr= SVR()

#entreno el modelo
svr.fit(X_train,y_train)

#Realizo una prediccion
Y_predic = svr.predict(X_test)

#graficamos los datos junto al modelo
plt.scatter(X_test,y_test)
plt.plot(X_test,Y_predic, color='red', linewidth=3)
plt.show()























