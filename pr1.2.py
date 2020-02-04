import numpy as np
import matplotlib.pyplot as plt


epsilon=0.001
r=3.25
x0=0.5
delta_x0 = 0.01
delta_r = 0.001

def logistica(x):
    return r*x*(1-x)

def orbita(x0,f,n):
    orbita=[]
    x = x0
    for i in range(n):
        x = f(x)
        orbita.append(x)
    return orbita
#arange(start,stop,step)
for x0 in np.arange(0, 1 + delta_x0, delta_x0):
    for r in np.arange(3 + delta_r, 4, delta_r):
        M = 10000
        N = 20
        orb = orbita(x0, logistica, M)
        posiblesk = [i for i in range(M-20, M)]
        V0 = []
        #Calcular tambien la cota de error de los vi del V0
        #Quedarse con el noveno mas grande (el percentil 90)
        #Como no es gaussiano hay que estimarlo
        equivalentesk = []
        for k in reversed(posiblesk):
            if k not in equivalentesk :
                kbueno = False
                for i in reversed(posiblesk):
                    if i != k and i not in equivalentesk and abs(orb[k] - orb[i]) < epsilon:
                        kbueno = True
                        equivalentesk.append(i)
                if kbueno : V0.append(k)
                equivalentesk.append(k)
            
       # if V0 == [] : print('No hay ks')
        
        if len(V0) == 8 : 
            print('x0: ', x0, ' r: ', r)
            print('V0: ', V0)
            #Siempre sale V0 con 1, 2 o 4 elementos (1 cuando x0 = 0 o 1)
plt.show()






























