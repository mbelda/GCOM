# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 23:53:36 2020

@author: robert monjo
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from numpy import trapz
#https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html

os.getcwd()

#q = variable de posición, dq0 = \dot{q}(0) = valor inicial de la derivada
#d = granularidad del parámetro temporal
def deriv(q,dq0,d):
   #dq = np.empty([len(q)])
   dq = (q[1:len(q)]-q[0:(len(q)-1)])/d
   dq = np.insert(dq,0,dq0) #dq = np.concatenate(([dq0],dq))
   return dq

#Ecuación de un sistema dinámico continuo
#Ejemplo de oscilador simple
def F(q):
    ddq = - 2*q*(q**2 - 1)
    return ddq

#Resolución de la ecuación dinámica \ddot{q} = F(q), obteniendo la órbita q(t)
#Los valores iniciales son la posición q0 := q(0) y la derivada dq0 := \dot{q}(0)
def orb(n,q0,dq0,F, args=None, d=0.001):
    #q = [0.0]*(n+1)
    q = np.empty([n+1])
    q[0] = q0
    q[1] = q0 + dq0*d
    for i in np.arange(2,n+1):
        args = q[i-2]
        q[i] = - q[i-2] + d**2*F(args) + 2*q[i-1]
    return q #np.array(q),

def periodos(q,d,max=True):
    #Si max = True, tomamos las ondas a partir de los máximos/picos
    #Si max == False, tomamos los las ondas a partir de los mínimos/valles
    epsilon = 5*d
    dq = deriv(q,dq0=None,d=d) #La primera derivada es irrelevante
    if max == True:
        waves = np.where((np.round(dq,int(-np.log10(epsilon))) == 0) & (q >0))
    if max != True:
        waves = np.where((np.round(dq,int(-np.log10(epsilon))) == 0) & (q <0))
    diff_waves = np.diff(waves)
    waves = waves[0][1:][diff_waves[0]>1]
    pers = diff_waves[diff_waves>1]*d
    return pers, waves

#################################################################    
#  CÁLUCLO DE ÓRBITAS
################################################################# 

#Ejemplo gráfico del oscilador simple
q0 = 0.
dq0 = 1.
fig, ax = plt.subplots(figsize=(12,5))
plt.ylim(-1.5, 1.5)  
plt.rcParams["legend.markerscale"] = 6
ax.set_xlabel("t = n $\delta$", fontsize=12)
ax.set_ylabel("q(t)", fontsize=12)
iseq = np.array([1,1.1,1.5,1.8,3])
for i in iseq:
    d = 10**(-i)
    n = int(32/d)
    t = np.arange(n+1)*d
    q = orb(n,q0=q0,dq0=dq0,F=F,d=d)
    plt.plot(t, q, 'ro', markersize=0.5/i,label='$\delta$ ='+str(np.around(d,3)), \
                 c=plt.get_cmap("winter")(i/np.max(iseq)))
    ax.legend(loc=3, frameon=False, fontsize=12)
#plt.savefig('Time_granularity.png', dpi=250)

#Ejemplo de coordenadas canónicas (q, p)
#Nos quedamos con el más fino y calculamos la coordenada canónica 'p'
q0 = 0.
dq0 = 1.
d = 10**(-3)
n = int(32/d)
t = np.arange(n+1)*d
q = orb(n,q0=q0,dq0=dq0,F=F,d=d)
dq = deriv(q,dq0=dq0,d=d)
p = dq/2

#Ejemplo gráfico de la derivada de q(t)
fig, ax = plt.subplots(figsize=(12,5))
plt.ylim(-1.5, 1.5)  
plt.rcParams["legend.markerscale"] = 6
ax.set_xlabel("t = n $\delta$", fontsize=12)
ax.set_ylabel("dq(t)", fontsize=12)
plt.plot(t, dq, '-')

#Ejemplo de diagrama de fases (q, p)
fig, ax = plt.subplots(figsize=(5,5))
plt.xlim(-1.1, 1.1)  
plt.ylim(-1, 1) 
plt.rcParams["legend.markerscale"] = 6
ax.set_xlabel("q(t)", fontsize=12)
ax.set_ylabel("p(t)", fontsize=12)
plt.plot(q, p, '-')
plt.show()

#################################################################    
#  ESPACIO FÁSICO
################################################################# 

## Pintamos el espacio de fases
def simplectica(q0,dq0,F,col=0,d = 10**(-4),n = int(16/d),marker='-'): 
    q = orb(n,q0=q0,dq0=dq0,F=F,d=d)
    dq = deriv(q,dq0=dq0,d=d)
    p = dq/2
    plt.plot(q, p, marker,c=plt.get_cmap("winter")(col))

#Dibuja el espacio de fases para las condiciones iniciales en los rangos por parámetro
def dibujarEspacioFases(q0Min, q0Max, dq0Min, dq0Max):
    fig = plt.figure(figsize=(8,5))
    fig.subplots_adjust(hspace=0.4, wspace=0.2)
    seq_q0 = np.linspace(q0Min, q0Max, num=12)
    seq_dq0 = np.linspace(dq0Min, dq0Max, num=12)
    for i in range(len(seq_q0)):
        for j in range(len(seq_dq0)):
            q0 = seq_q0[i]
            dq0 = seq_dq0[j]
            ax = fig.add_subplot(1,1, 1)
            col = (1+i+j*(len(seq_q0)))/(len(seq_q0)*len(seq_dq0))
            #ax = fig.add_subplot(len(seq_q0), len(seq_dq0), 1+i+j*(len(seq_q0)))
            simplectica(q0=q0,dq0=dq0,F=F,col=col,marker='ro',d= 10**(-3),n = int(16/d))
    ax.set_xlabel("q(t)", fontsize=12)
    ax.set_ylabel("p(t)", fontsize=12)
    #fig.savefig('Simplectic.png', dpi=250)
    plt.show()

##Espacio fásico con condiciones iniciales [0,1]x[0,1]
dibujarEspacioFases(0., 1., 0., 2.)

#################################################################    
#  CÁLCULO DEL ÁREA DEL ESPACIO FÁSICO
#################################################################  
#Tomamos un par (q(0), p(0)) y nos quedamos sólo en un trozo/onda de la órbita, sin repeticiones
#Para eso, tomaremos los periodos de la órbita, que definen las ondas

def calculaArea(q0, dq0, delta):
    d = delta
    n = int(32/d)
    t = np.arange(n+1)*d
    q = orb(n,q0=q0,dq0=dq0,F=F,d=d)
    dq = deriv(q,dq0=dq0,d=d)
    p = dq/2
    fig, ax = plt.subplots(figsize=(12,5))
    plt.plot(t, q, '-')
    
    #Nos aseguramos que tiene buen aspecto
    fig, ax = plt.subplots(figsize=(5,5)) 
    plt.rcParams["legend.markerscale"] = 6
    ax.set_xlabel("q(t)", fontsize=12)
    ax.set_ylabel("p(t)", fontsize=12)
    plt.plot(q, p, '-')
    plt.show()
    
    #Tomaremos los periodos de la órbita, que definen las ondas
    T, W = periodos(q,d,max=False)
    #Nos quedamos con el primer trozo
    plt.plot(q[W[0]:W[1]])
    plt.show()
    plt.plot(p[W[0]:W[1]])
    plt.show()
    plt.plot(q[W[0]:W[1]],p[W[0]:W[1]])
    plt.show()
    
    
    #Tomamos la mitad de la "curva cerrada" para integrar más fácilmente
    mitad = np.arange(W[0],W[0]+np.int((W[1]-W[0])/2),1)
    plt.plot(q[mitad],p[mitad])
    plt.show()
    
    # Regla del trapezoide
    return 2*trapz(p[mitad],q[mitad])

#Calcula el área sin hacer los dibujos de comprobación
def calculaAreaSinDibujar(q0, dq0, delta):
    d = delta
    n = int(32/d)
    q = orb(n,q0=q0,dq0=dq0,F=F,d=d)
    dq = deriv(q,dq0=dq0,d=d)
    p = dq/2

    #Tomaremos los periodos de la órbita, que definen las ondas
    T, W = periodos(q,d,max=False)
    if W.size < 2:
        return - W.size
    else:
        #Tomamos la mitad de la "curva cerrada" para integrar más fácilmente
        mitad = np.arange(W[0],W[0]+np.int((W[1]-W[0])/2),1)
        
        # Regla del trapezoide
        return 2*trapz(p[mitad],q[mitad])

#Calcula las condiciones iniciales que maximizan y minimizan el área.
#Además, devuelve la correspondiente área máxima y mínima
def calculaExtremos(q0Min, q0Max, dq0Min, dq0Max, delta):
    areaMax = 0.
    qpMax = [0.,0.]
    areaMin = 30.
    qpMin = [0.,0.]
    paso = 0.1
    for q0 in np.arange(q0Min, q0Max + paso, paso):
        for dq0 in np.arange(dq0Min, dq0Max + paso, paso):
            area = calculaAreaSinDibujar(q0, dq0, delta)
            if area > 0: 
                if area > areaMax:
                    areaMax = area
                    qpMax = [q0,dq0]
                if area < areaMin:
                    areaMin = area
                    qpMin = [q0,dq0]
            
    return areaMax, qpMax, areaMin, qpMin           

#Vamos a calcular el error variando delta
def calculaErrores(deltaMin, deltaMax, areaRef, qpMin, qpMax):
    errores = np.array([])
    for deltaAux in np.arange(deltaMin,deltaMax,0.05):
        areaAuxMax = calculaAreaSinDibujar(qpMax[0], qpMax[1], 10**(-deltaAux))
        areaAuxMin = calculaAreaSinDibujar(qpMin[0], qpMin[1], 10**(-deltaAux))
        if areaAuxMin > 0 and areaAuxMax > 0:
            areaAux = areaAuxMax - areaAuxMin
            errores = np.append(errores, areaRef - areaAux)
    
    errores = np.abs(errores)
    return errores


delta = 10**(-3.5)
#Calculamos el área para D0 por tanto q esta en [0,1] y dq en [0,2]
areaMax, qpMax, areaMin, qpMin = calculaExtremos(0. ,1. , 0. ,2. , delta)
area = areaMax - areaMin
error = np.amax(calculaErrores(3, 3.5, area, qpMin, qpMax))
print("El área total del espacio de fases con condiciones iniciales D0 es: " + str(area))
print("Con la órbita de área máxima en condiciones iniciales " + str(qpMax) \
        + " y la mínima en " + str(qpMin))
print("Con un error de: " + str(error))
