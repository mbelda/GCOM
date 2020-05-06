# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 11:06:03 2020

@author: Majo
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D

alpha = 10.0
beta = 28.0
rho = 8.0 / 3.0

#Ecuaciones de Lorenz
def f(estado, t):
    q1, q2, q3 = estado
    return alpha * (q2 - q1), beta*q1 - q2 - q1*q3, q1*q2 - rho*q3

#Calcula las derivadas parciales de las tres variables
def derivParcial(q,estado0,d):
   dq10, dq20, dq30 = estado0
   q = np.transpose(q)
   nCols = len(q[0])
   dq1 = (q[0,1:nCols]-q[0, 0:(nCols-1)])/d
   dq1 = np.insert(dq1, 0, dq10) #dq = np.concatenate(([dq0],dq))
   dq2 = (q[1,1:nCols]-q[1, 0:(nCols-1)])/d
   dq2 = np.insert(dq2, 0, dq20) #dq = np.concatenate(([dq0],dq))
   dq3 = (q[2,1:nCols]-q[2, 0:(nCols-1)])/d
   dq3 = np.insert(dq3, 0, dq30) #dq = np.concatenate(([dq0],dq))
   
   dq = np.array([dq1, dq2, dq3])
   return np.transpose(dq)


#Discretizamos el sistema
d = 0.01
t = np.arange(0.0, 100.0, d)

#Fijamos las condiciones iniciales
estadoInicial = [1.0, 1.0, 1.0]

#Calculamos la órbita del sistema para las condiciones iniciales dadas
q = odeint(f, estadoInicial, t)
p = derivParcial(q,estadoInicial,d)/2

#Representamos q
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(q[:, 0], q[:, 1], q[:, 2])
plt.show()

#Representamos p
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(p[:, 0], p[:, 1], p[:, 2])
plt.show()


#Calculamos el espacio fásico para las condiciones iniciales [-1,1]x[-1,1]x[-1,1]

list_q = []
list_p = []

pasoCondIni = 0.25
# Con los valores actuales de t, i0, j0 y k0 tarda bastante en ejecutarse 
for i0 in np.arange(-1. ,1. + pasoCondIni ,pasoCondIni):
    for j0 in np.arange(-1. ,1. + pasoCondIni ,pasoCondIni):
        for k0 in np.arange(-1. ,1. + pasoCondIni ,pasoCondIni):
            estadoInicial = [i0,j0,k0]
            nuevo_q = odeint(f,estadoInicial,t)
            list_q.append(nuevo_q)
            list_p.append(derivParcial(nuevo_q,estadoInicial,d)/2)

#Representamos el espacio fásico en tres gráficas
for i in range(3):
    plt.figure()
    for j in range(len(list_q)):
        plt.plot(list_q[j][:,i], list_p[j][:,i], '-',c=plt.get_cmap("winter")(0))
    plt.show()














