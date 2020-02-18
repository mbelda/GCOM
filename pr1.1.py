import numpy as np
import matplotlib.pyplot as plt
import operator


epsilon = 0.001
r_inicial = 3.1
x0_inicial = 0.2
delta_x0 = 0.05
delta_r = 0.001
M = 100
N = 20

def resta(x,y):
    lista_resta = list(map(operator.sub, x, y))
    return list(map(abs, lista_resta))

def logistica(x, r):
    return r*x*(1-x)

def orbita(x0, f, n, r):
    orbita=[]
    x = x0
    for i in range(n):
        x = f(x, r)
        orbita.append(x)
    return orbita

def calculaV0(x0, r):
    orb = orbita(x0, logistica, M, r)
    posiblesk = [i for i in range(M-N, M)]
    V0 = []
    V0orb = []
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
    return V0, V0orb
 
def calculaError(V0orb, r) :
    errores = []
    V0orb.sort()   
    Vaux = V0orb.copy()
    for j in range(0, 10):
        for i in range(len(V0)):
            Vaux[i] = logistica(Vaux[i], r)
        Vaux.sort()   
        errores.append(max(resta(V0orb, Vaux)))
    errores.sort()
    return errores[8]

def estabilidad(x0, r):
    for x0 in np.arange(x0, x0 + delta_x0, delta_x0/10):
        V0aux, V0auxorb = calculaV0(x0, r)    
        V0auxorb.sort()
        if max(resta(V0orb,V0auxorb)) > epsilon:
            return False
    return True

def bifurcaciones(nelems, x0):
    V0s = np.array([])
    enc = False
    for r in np.arange(2.5 + delta_r, 4, delta_r):
        V0aux, V0auxorb = calculaV0(x0, r)   
        V0auxorb.sort()
        V0auxorb = np.array(V0auxorb)
        np.append(V0s, V0auxorb)
        
        for y in V0auxorb:
            plt.plot(r, y, ',k', alpha=0.25)
            
        if len(V0aux) == nelems and enc == False:
            enc = True
            print('En r = ' + str(r) + ' V0 tiene ' + str(nelems) + ' elementos')

print('Fijamos x0 = 0.2')
V0, V0orb = calculaV0(x0_inicial, r_inicial)
print('Un conjunto atractor con r = 3.1 es: ' + str(V0orb))
print('su error es:', calculaError(V0orb, r_inicial))

if estabilidad(x0_inicial, r_inicial) == True:
    print('Es estable')
else :
    print('No es estable')

V0, V0orb = calculaV0(x0_inicial, 3.4)
print('Otro conjunto atractor con r = 3.4 es: ' + str(V0orb))
print('su error es:', calculaError(V0orb, 3.4))

if estabilidad(x0_inicial, 3.4) == True:
    print('Es estable')
else :
    print('No es estable')

bifurcaciones(8, x0_inicial)       
plt.show()






























