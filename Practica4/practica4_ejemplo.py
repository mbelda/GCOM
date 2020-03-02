# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 18:55:58 2020

@author: Robert
"""

import sklearn as sk  #PCA
import seaborn as sns #Dataset de ejemplo
import scipy as sc    #SVD
import numpy as np
from numpy.linalg import multi_dot as dot  #Operaciones matriciales
import matplotlib.pyplot as plt

# Factorización con svd
# svd factoriza la matriz A  en dos matrices unitarias U y Vh, y una 
# matriz s de valores singulares (reales, no negativo) de tal manera que
# A == U * S * Vh, donde S es una matriz con s como principal diagonal y ceros

A = np.array([[2, 4]
             ,[1, 3]
             ,[0, 0]
             ,[0, 0]])
print(A.shape)

#Vh = La matriz unitaria
#U = La matriz unitaria
#s = Valores singulares
U, s, Vh = sc.linalg.svd(A)

print(U.shape, Vh.shape, s.shape)

# Generando S
S = sc.linalg.diagsvd(s, 4, 2)

# Reconstruyendo la Matriz A.
A2 = dot([U, S, Vh])

## IMPORTANT!!!!!!!!
##https://relopezbriega.github.io/blog/2016/09/13/factorizacion-de-matrices-con-python/

iris = sns.load_dataset("iris")
print(iris.shape)
iris.head()

# Ejemplo de PCA con Scikit-Learn e Iris dataset

# Divido el dataset en datos y clases
X = iris.ix[:,0:4].values
y = iris.ix[:,4].values

# Estandarizo los datos
X_std = sk.preprocessing.StandardScaler().fit_transform(X)

pca = sk.decomposition.PCA(n_components=2)
Y_pca = pca.fit_transform(X_std)

# Visualizo el resultado
for lab, col in zip(('setosa', 'versicolor', 'virginica'),
                        ('blue', 'red', 'green')):
    plt.scatter(Y_pca[y==lab, 0],
                    Y_pca[y==lab, 1],
                    label=lab,
                    c=col)
    
plt.xlabel('Componente 1')
plt.ylabel('Componente 2')
plt.legend(loc='lower center')
plt.tight_layout()
plt.title('Ejemplo PCA')
plt.show()