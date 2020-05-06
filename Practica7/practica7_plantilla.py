# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 18:58:33 2020

@author: Robert
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import animation
from skimage import io, color

vuestra_ruta = r"C:\Users\Majo\Documents\GitHub\GCOM\Practica7"

os.getcwd()
os.chdir(vuestra_ruta)


"""
Ejemplo para el apartado 1.

Modifica la figura 3D y/o cambia el color
https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
"""
def getCilindro():
   ro = np.linspace(0, 10, 25)
   phi = np.linspace(0, 2*np.pi, 100)
   Ro, Phi = np.meshgrid(ro, phi)

   X = 3 * np.cos(Phi)
   Y = Ro
   Z = 3 * np.sin(Phi)

   return X, Y, Z
 

fig = plt.figure()
ax = plt.axes(projection='3d')
X, Y, Z = getCilindro()
ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
#cset = ax.contour(X, Y, Z, 16, extend3d=True,cmap = plt.cm.get_cmap('viridis'))
#ax.clabel(cset, fontsize=9, inline=1)
plt.show()


"""
Transformación para el segundo apartado

NOTA: Para el primer aparado es necesario adaptar la función o crear otra similar
pero teniendo en cuenta más dimensiones
"""

def transf1D(x,y,z,M, v=np.array([0,0,0])):
    xt = np.empty(len(x))
    yt = np.empty(len(x))
    zt = np.empty(len(x))
    centroide = calcCentroide1(x,y)
    for i in range(len(x)):
        q = np.array([x[i],y[i],z[i]])
        q[0] = q[0] - centroide[0]
        q[1] = q[1] - centroide[1]
        xt[i], yt[i], zt[i] = np.matmul(M, q) + v
    return xt, yt, zt


def transf2D(x,y,z,M, v=np.array([0,0,0])):
    nFilas, nCols = x.shape
    xt2 = np.empty((nFilas,nCols))
    yt2 = np.empty((nFilas,nCols))
    zt2 = np.empty((nFilas,nCols))
    centroide = calcCentroide2(x,y,z)
    for i in range(nFilas):
        for j in range(nCols):
            q = np.array([x[i][j],y[i][j],z[i][j]])
            xt2[i][j], yt2[i][j], zt2[i][j] = np.matmul(M, q-centroide) + v
    return xt2, yt2, zt2
    
def calcCentroide2(X,Y,Z):
    nFilas, nCols = X.shape
    xC = np.sum(X)/(nFilas*nCols)
    yC = np.sum(Y)/(nFilas*nCols)
    zC = np.sum(Z)/(nFilas*nCols)
    return xC, yC, zC
    
def calcCentroide1(x,y):
    nElems = len(x)
    xC = np.sum(x)/nElems
    yC = np.sum(y)/nElems
    return xC, yC

def calcDiam3(X,Y,Z):
    nFilas, nCols = X.shape
    max = 0
    for i1 in range(X.shape[0]):
        for j1 in range(X.shape[1]):
            for i2 in range(X.shape[0] - i1):
                for j2 in range(X.shape[1] - j1):
                    dist = (X[i1,j1]-X[i2,j2])**2 + (Y[i1,j1]-Y[i2,j2])**2 + (Z[i1,j1]-Z[i2,j2])**2
                    if dist > max:
                        max = dist
    return np.sqrt(max)
    
    
def calcDiam2(X,Y):
    nElems = len(X)
    max = 0
    for i1 in range(0,nElems,4):
        print(i1)
        for i2 in range(0,nElems - i1,4):
            dist = (X[i1]-X[i2])**2 + (Y[i1]-Y[i2])**2
            if dist > max:
                max = dist
    return np.sqrt(max)
    
# def calcDiam(xMax, xMin, yMax, yMin):
    # return np.sqrt((xMax - xMin)**2 + (yMax - yMin)**2)


def animate2D(t,diam):
    theta = 3*np.pi*t
    M = np.array([[np.cos(theta),- np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]])
    
    ax = plt.axes(xlim=(-5,20), ylim=(-5,20), zlim=(-5,20),projection='3d')
    #ax.view_init(60, 30)
    
    v = np.array([diam, diam, 0])*t

    Xt, Yt, Zt = transf2D(X, Y, Z, M=M, v=v)
    ax.plot_surface(Xt, Yt, Zt, rstride=1, cstride=1,cmap='viridis', edgecolor='none') 
    return ax,
    

#diametro3 = calcDiam(X.max(),X.min(),Y.max(),Y.min())
diametro3 = calcDiam3(X,Y,Z)
print('diámetro',diametro3)

def init2D():
    return animate2D(0,diametro3),

animate2D(np.arange(0.1, 1,0.1)[5],diametro3)
plt.show()


fig = plt.figure(figsize=(6,6))
ani = animation.FuncAnimation(fig, animate2D, frames=np.arange(0,1,0.025), init_func=init2D, fargs=[diametro3],
                              interval=20)
os.chdir(vuestra_ruta)
ani.save("p7a.gif", fps = 10)  
os.getcwd()

"""
Segundo apartado casi regalado

Imagen del árbol
"""

os.getcwd()
os.chdir(vuestra_ruta)

img = io.imread('arbol.png')
dimensions = color.guess_spatial_dimensions(img)
print(dimensions)
io.show()
#io.imsave('arbol2.png',img)

#https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
fig = plt.figure(figsize=(5,5))
p = plt.contourf(img[:,:,0],cmap = plt.cm.get_cmap('viridis'), levels=np.arange(0,240,2))
plt.axis('off')
#fig.colorbar(p)

xyz = img.shape

x = np.arange(0,xyz[0],1)
y = np.arange(0,xyz[1],1)
xx,yy = np.meshgrid(x, y)
xx = np.asarray(xx).reshape(-1)
yy = np.asarray(yy).reshape(-1)
z  = img[:,:,0]
zz = np.asarray(z).reshape(-1)


"""
Consideraremos sólo los elementos con zz < 240 

Por curiosidad, comparamos el resultado con contourf y scatter!
"""
#Variables de estado coordenadas
x0 = xx[zz<240]
y0 = yy[zz<240]
z0 = zz[zz<240]/256.


#Variable de estado: color
col = plt.get_cmap("viridis")(np.array(0.1+z0))

fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(1, 2, 1)
plt.contourf(x,y,z,cmap = plt.cm.get_cmap('viridis'), levels=np.arange(0,240,2))
ax = fig.add_subplot(1, 2, 2)
plt.scatter(x0,y0,c=col,s=0.1)
plt.show()



def animate(t,diam):
    theta = 3*np.pi*t
    M = np.array([[np.cos(theta),- np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]])
    v = np.array([diam, diam, 0])*t
    
    ax = plt.axes(xlim=(-100,600), ylim=(-100,600), projection='3d')
    #ax.view_init(60, 30)

    XYZ = transf1D(x0, y0, z0, M=M, v=v)
    col = plt.get_cmap("viridis")(np.array(0.1+XYZ[2]))
    ax.scatter(XYZ[0],XYZ[1],c=col,s=0.1,animated=True)
    return ax,

#diametro2 = calcDiam(x0.max(),x0.min(),y0.max(),y0.min())
diametro2 = calcDiam2(x0,y0)
print('diámetro', diametro2)

def init():
    return animate(0,diametro2),

animate(0,diametro2)
plt.show()


fig = plt.figure(figsize=(6,6))
ani = animation.FuncAnimation(fig, animate, frames=np.arange(0,1,0.025), init_func=init, fargs=[diametro2],
                              interval=20)
os.chdir(vuestra_ruta)
ani.save("p7b.gif", fps = 10)  
os.getcwd()







