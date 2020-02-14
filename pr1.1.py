import numpy as np
import matplotlib.pyplot as plt


epsilon=0.001
r = 3.4
x0 = 0.2
delta_x0 = 0.05
delta_r = 0.001

def resta(x,y):
    res = []
    for i in range(len(x)):
        res.append(np.abs(x[i] - y[i]))
    return res

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

M = 100
N = 20
orb = orbita(x0, logistica, M)
posiblesk = [i for i in range(M-20, M)]
V0 = []
V0orb = []
#Calcular tambien la cota de error de los vi del V0
#Quedarse con el noveno mas grande (el percentil 90)
#Como no es gaussiano hay que estimarlo
equivalentesk = []
for k in reversed(posiblesk):
    if k not in equivalentesk :
        kbueno = False
        for i in range(k):
            if i not in equivalentesk and abs(orb[k] - orb[k-i-1]) < epsilon:
                kbueno = True
                equivalentesk.append(k-i-1)
        if kbueno :
            V0.append(k)
            V0orb.append(orb[k])
        equivalentesk.append(k)
    
if V0 == [] : print('No hay ks')
errores = []
V0orb.sort()   
Vaux = V0orb
for j in range(0, 10):
    for i in range(len(V0)):
        Vaux[i] = logistica(Vaux[i])
    Vaux.sort()    
    errores.append(max(resta(V0orb, Vaux)))

errores.sort()  
print(errores)
print('El error es:', errores[8])
print('V0: ', V0)


for x0 in np.arange(x0, x0 + delta_x0, delta_x0/10):
    M = 100
    N = 20
    orb = orbita(x0, logistica, M)
    posiblesk = [i for i in range(M-20, M)]
    V0aux = []
    V0auxorb = []
    #Calcular tambien la cota de error de los vi del V0
    #Quedarse con el noveno mas grande (el percentil 90)
    #Como no es gaussiano hay que estimarlo
    equivalentesk = []
    for k in reversed(posiblesk):
        if k not in equivalentesk :
            kbueno = False
            for i in range(k):
                if i not in equivalentesk and abs(orb[k] - orb[k-i-1]) < epsilon:
                    kbueno = True
                    equivalentesk.append(k-i-1)
            if kbueno :
                V0aux.append(k)
                V0auxorb.append(orb[k])
            equivalentesk.append(k)
    
    V0auxorb.sort()
    print(V0orb)
    print(V0auxorb)
    if max(resta(V0orb,V0auxorb)) > epsilon:
        print('No es estable')
V0s = np.array([])
#np.linspace(start, stop, numintervals)
enc = False
for r in np.arange(2.5 + delta_r, 4, delta_r):
    M = 100
    N = 20
    orb = orbita(x0, logistica, M)
    posiblesk = [i for i in range(M-20, M)]
    V0aux = []
    V0auxorb = []
    #Calcular tambien la cota de error de los vi del V0
    #Quedarse con el noveno mas grande (el percentil 90)
    #Como no es gaussiano hay que estimarlo
    equivalentesk = []
    for k in reversed(posiblesk):
        if k not in equivalentesk :
            kbueno = False
            for i in range(k):
                if i not in equivalentesk and abs(orb[k] - orb[k-i-1]) < epsilon:
                    kbueno = True
                    equivalentesk.append(k-i-1)
            if kbueno :
                V0aux.append(k)
                V0auxorb.append(orb[k])
            equivalentesk.append(k)
    
    V0auxorb.sort()
    V0auxorb = np.array(V0auxorb)
    np.append(V0s, V0auxorb)
    
    for y in V0auxorb:
        plt.plot(r, y, ',k', alpha=0.25)
        
    if len(V0aux) == 8 and enc == False:
        enc = True
        print('Los elementos son 8 a partir de r =', r)

        
plt.show()






























