# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 12:33:47 2020

@author: Majo
"""
import numpy as np

iteraciones = 50
epsilon = 1e-40

def H(d):
    #lim delta -> 0
    suma = 0
    i = iteraciones
    delta = (1/3)**i
    #suma normas 1 ^d
    suma = 8**i * delta**2**d
    
    return suma
    
d = 1.99
while H(d) < epsilon :
    d = d - 1e-10
    print(d)

print(d)
print('Log8/log3 =', np.log(8)/np.log(3))