import numpy as np
import matplotlib.pyplot as plt
import operator

#Variables iniciales
epsilon = 0.001
r_inicial = 3.1
x0_inicial = 0.2
delta_x0 = 0.05
delta_r = 0.001
M = 100
N = 20

#Resta con valor absoluto dos listas
def resta(x,y):
    lista_resta = list(map(operator.sub, x, y))
    return list(map(abs, lista_resta))

#Función logistica que define el sistema
def logistica(x, r):
    return r*x*(1-x)

#Calcula n términos de la sucesión de la órbita
#de una función f por un punto x0
def orbita(x0, f, n, r):
    orbita=[]
    x = x0
    for i in range(n):
        x = f(x, r)
        orbita.append(x)
    return orbita

#Calcula el conunto límite de puntos atractores dado un x0
def calculaV0(x0, r):
    orb = orbita(x0, logistica, M, r)
    posiblesk = [i for i in range(M-N, M)]
    #Guardamos en V0 los valores k de los limites
    V0 = []
    #Guardamos el valor de la orbita del limite
    V0orb = []
    #En equivalentesk guardamos los valores de las que ya
    #tenemos un representante de ese limite
    equivalentesk = []
    #Recorremos los candidatos
    for k in reversed(posiblesk) :
        #Si no es equivalente a uno de los valores limite encontrados
        if k not in equivalentesk :
            #Suponemos que no será limite
            kbueno = False
            for i in range(k) :
                #Recorremos los puntos que quedan
                if i not in equivalentesk and abs(orb[k] - orb[k-i-1]) < epsilon:
                    #Hay algún punto cercano a él, luego k es limite
                    kbueno = True
                    #Ademas, el k-i-1 es equivalente a k pues
                    #los valores de la orbita son muy cercanos
                    equivalentesk.append(k-i-1)
            if kbueno :
                #Como es valor limite, lo añadimos a las listas
                V0.append(k)
                V0orb.append(orb[k])
            equivalentesk.append(k)
    return V0, V0orb

#Calcula el error cometido al aproximar los valores limites con V0
def calculaError(V0orb, r) :
    errores = []
    V0orb.sort()   
    Vaux = V0orb.copy()
    for j in range(0, 10):
        for i in range(len(V0)):
            Vaux[i] = logistica(Vaux[i], r)
        Vaux.sort()
        #Tomamos el máximo de las diferencias de las órbitas de
        #los índices de V0 y los V anteriores a él
        errores.append(max(resta(V0orb, Vaux)))
    errores.sort()
    #Tomamos el percentil noventa de los diez errores anteriores
    return errores[8]

#Comprueba que el sistema sea estable para el x0 dado por parámetro
def estabilidad(x0, r):
    #Variamos ligeramente x0
    for x0 in np.arange(x0, x0 + delta_x0, delta_x0/10):
        V0aux, V0auxorb = calculaV0(x0, r)    
        V0auxorb.sort()
        #Comprobamos que los valores del conjunto limite esten cercanos
        if max(resta(V0orb,V0auxorb)) > epsilon:
            return False
    return True

#Calcula el r a partir del cual hay conjuntos limite de al menos nelems
#y dibuja la grafica de bifurcaciones para r entre 2.5 y 4
def bifurcaciones(nelems, x0):
    V0s = np.array([])
    enc = False
    #Variamos r entre 2.5 y 4
    for r in np.arange(2.5 + delta_r, 4, delta_r):
        V0aux, V0auxorb = calculaV0(x0, r)   
        V0auxorb.sort()
        V0auxorb = np.array(V0auxorb)
        np.append(V0s, V0auxorb)
        
        #dibujamos la grafica de las bifurcaciones del sistema
        for y in V0auxorb:
            plt.plot(r, y, ',k', alpha=0.25)
        
        #Vemos si es el primer r en alcanzar los nelems
        if len(V0aux) == nelems and enc == False:
            enc = True
            print('En r = ' + str(r) + ' V0 tiene ' + str(nelems) + ' elementos')


#MAIN
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






























