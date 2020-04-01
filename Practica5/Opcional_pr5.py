# -*- coding: utf-8 -*-
"""
APARTADO 3
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


#Dominio
u = np.linspace(0, np.pi, 25)
v = np.linspace(0, 2 * np.pi, 50)

#Esfera
x = np.outer(np.sin(u), np.sin(v))
y = np.outer(np.sin(u), np.cos(v))
z = np.outer(np.cos(u), np.ones_like(v))

#Curva
t2 = np.linspace(0.001, 1, 200)
x2 = abs(t2) * np.sin(30 * t2/2)
y2 = abs(t2) * np.cos(30 * t2/2)
z2 = np.sqrt(1-x2**2-y2**2)
c2 = x2 + y2

def animate2(t):
    eps = 1e-16
    
    #Esfera
    xt = 1/((1-t) + np.abs(np.tan(-1) - np.tan(z))*t + eps)*x
    yt = 1/((1-t) + np.abs(np.tan(-1) - np.tan(z))*t + eps)*y
    zt = -t + z*(1-t)
    
    #Curva
    x2t = 1/((1-t) + np.abs(np.tan(-1) - np.tan(z2))*t + eps)*x2
    y2t = 1/((1-t) + np.abs(np.tan(-1) - np.tan(z2))*t + eps)*y2
    z2t = -t + z2*(1-t)
    
    ax = plt.axes(projection='3d')
    ax.set_xlim3d(-1,1)
    ax.set_ylim3d(-1,1)
    ax.set_zlim3d(-1,1)
    ax.plot_surface(xt, yt, zt, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.plot(x2t,y2t, z2t, '-b', c="black", zorder=3)
    return ax,

def init2():
    return animate2(0),

animate2(np.arange(0, 1,0.1)[1])
plt.show()

fig = plt.figure(figsize=(6,6))
ani = animation.FuncAnimation(fig, animate2, np.arange(0,1,0.05), init_func=init2,
                              interval=20)
ani.save("miejemplo.gif", fps = 5)



##Version 2
def animate(t):
    eps = 1e-16
    
    #Esfera
    xt = 1/(np.tan(np.pi/4 - t) + np.abs(-1-z)*np.tan(t) + eps)*x
    yt = 1/(np.tan(np.pi/4 - t) + np.abs(-1-z)*np.tan(t) + eps)*y
    zt = np.tan(-t) + z*np.tan(np.pi/4 - t)
    

    #Curva
    x2t = 1/(np.tan(np.pi/4 - t) + np.abs(-1-z2)*np.tan(t) + eps)*x2
    y2t = 1/(np.tan(np.pi/4 - t) + np.abs(-1-z2)*np.tan(t) + eps)*y2
    z2t = np.tan(-t) + z2*np.tan(np.pi/4 - t)
    
    ax = plt.axes(projection='3d')
    ax.set_xlim3d(-1,1)
    ax.set_ylim3d(-1,1)
    ax.set_zlim3d(-1,1)
    ax.plot_surface(xt, yt, zt, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.plot(x2t,y2t, z2t, '-b', c="black", zorder=3)
    return ax,

def init():
    return animate(0),

fig = plt.figure(figsize=(6,6))
ani = animation.FuncAnimation(fig, animate, np.arange(0, np.pi/4,0.05), init_func=init,
                              interval=20)
ani.save("mi2ejemplo.gif", fps = 5) 

