"""
Práctica 2
"""

import os
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

#### Carpeta donde se encuentran los archivos ####
ubica = r"C:\Users\Majo\Documents\GitHub\GCOM\Practica2"

#### Vamos al directorio de trabajo####
os.getcwd()
os.chdir(ubica)
#files = os.listdir(ubica)

with open('auxiliar_en_pract2.txt', 'r') as file:
      en = file.read()
     
with open('auxiliar_es_pract2.txt', 'r') as file:
      es = file.read()

#### Pasamos todas las letras a minúsculas
en = en.lower()
es = es.lower()

#### Contamos cuantas letras hay en cada texto
from collections import Counter
tab_en = Counter(en)
tab_es = Counter(es)

##### Transformamos en formato array de los carácteres (states) y su frecuencia
##### Finalmente realizamos un DataFrame con Pandas y ordenamos con 'sort'
tab_en_states = np.array(list(tab_en))
tab_en_weights = np.array(list(tab_en.values()))
tab_en_probab = tab_en_weights/float(np.sum(tab_en_weights))
distr_en = pd.DataFrame({'states': tab_en_states, 'probab': tab_en_probab})

#for i in range(len(distr_en)):
#    plt.plot(i, distr_en['probab'][i], '-ob')


distr_en = distr_en.sort_values(by='probab', ascending=True)
distr_en.index=np.arange(0,len(tab_en_states))

tab_es_states = np.array(list(tab_es))
tab_es_weights = np.array(list(tab_es.values()))
tab_es_probab = tab_es_weights/float(np.sum(tab_es_weights))
distr_es = pd.DataFrame({'states': tab_es_states, 'probab': tab_es_probab })

#for i in range(len(distr_es)):
#    plt.plot(i, distr_es['probab'][i], '-or')
    
distr_es = distr_es.sort_values(by='probab', ascending=True)
distr_es.index=np.arange(0,len(tab_es_states))


def huffman_branch(distr):
    states = np.array(distr['states'])
    probab = np.array(distr['probab'])
    state_new = np.array([''.join(states[[0,1]])])
    probab_new = np.array([np.sum(probab[[0,1]])])
    codigo = np.array([{states[0]: 0, states[1]: 1}])
    states =  np.concatenate((states[np.arange(2,len(states))], state_new), axis=0)
    probab =  np.concatenate((probab[np.arange(2,len(probab))], probab_new), axis=0)
    distr = pd.DataFrame({'states': states, 'probab': probab, })
    distr = distr.sort_values(by='probab', ascending=True)
    distr.index=np.arange(0,len(states))
    branch = {'distr':distr, 'codigo':codigo}
    return(branch) 

def huffman_tree(distr):
    tree = np.array([])
    while len(distr) > 1:
        branch = huffman_branch(distr)
        distr = branch['distr']
        code = np.array([branch['codigo']])
        tree = np.concatenate((tree, code), axis=None)
    return(tree)
 
#Generamos los arboles de huffman
tree_en = huffman_tree(distr_en)
tree_es = huffman_tree(distr_es)

#Codifica una letra 
def codifletra(letra, tree):
    codigo = ''
    #Recorremos los nodos persiguiendo la letra y guardando las ramas
    #por las que vamos, pues serán el codigo asociado
    for nodo in np.flip(tree):
        keys = np.array(list(nodo.keys()))
        values = np.array(list(nodo.values()))

        if letra in keys[0]:
            codigo = codigo + str(values[0])
        elif letra in keys[1]:
            codigo = codigo + str(values[1])
        
    return codigo

#Codifica un conjunto de caracteres
def codif(palabra, tree):
    codigo = ''
    for letra in list(palabra):
        codigo = codigo + codifletra(letra, tree)
        
    return codigo

#Busca el nodo hijo de otro nodo de un arbol que contiene la letra key
def buscaSig(key, count, tree):
    for i in range(count+1, len(tree)):
        nodo = np.flip(tree)[i]
        keys = np.array(list(nodo.keys()))
        #Vemos si la letra key esta la primera en el nodo izquierdo
        if keys[0][0] in key:
            return i

#Decodifica un codigo binario       
def decod(codigo, tree):
    palabra = ''
    count = 0
    #Para cada cifra del codigo tomamos la rama asociada hasta encontrar una hoja
    for i in codigo:
        nodo = np.flip(tree)[count]
        keys = np.array(list(nodo.keys()))
        #Si es una hoja
        if len(keys[int(i)]) == 1:
            palabra = palabra + keys[int(i)]
            count = 0
        else:
            #Bajamos al siguiente nodo correspondiente
            count = buscaSig(keys[int(i)], count, tree)
    return palabra

#Calcula el codigo huffman binario de un idioma, su longitud media (L)
#y su entropía (H)             
def huffmanS(tree, distr):
    dic = []
    L = 0
    H = 0
    for i in range(len(distr)):
        c = distr['states'][i]
        cod = codif(c, tree)
        L = L + len(cod)*distr['probab'][i]
        dic.append((c, cod))
        H = H - distr['probab'][i]*np.log2(distr['probab'][i])
    return dic, L, H

#Calcula el indice de Gini
def gini(distr):
    GI = 0
    for i in range(len(distr)):
        GI = GI + distr['probab'][i]/len(distr)
    return 1 - 2*GI

#Calcula el indice de diversidad de Hill
def hill(distr):
    hill = 0
    for i in range(len(distr)):
        hill = hill + distr['probab'][i]**2      
    return 1/hill


#MAIN
huf_en, L_en, H_en = huffmanS(tree_en, distr_en)
print('Codigo huffman ingles: ' + str(huf_en))
print('Longitud media: ' + str(L_en))
print('Entropía: ' + str(H_en))
print('Se cumple el teorema de Shannon, pues H <= L <= H+1')

huf_es, L_es, H_es = huffmanS(tree_es, distr_es)
print('Codigo huffman ingles: ' + str(huf_es))
print('Longitud media: ' + str(L_es))
print('Entropía: ' + str(H_es))
print('Se cumple el teorema de Shannon, pues H <= L <= H+1')

print('Codificacion de fractal en ingles: ' + codif('fractal', tree_en))
print('con longitud: ' + str(len(codif('fractal', tree_en))))
print('La longitud de fractal en ingles en binario usual es: ' + str(math.ceil(np.log2(len(distr_en)))*len('fractal')))
print('Codificacion de fractal en español: ' + codif('fractal', tree_es))
print('con longitud: ' + str(len(codif('fractal', tree_es))))
print('La longitud de fractal en español en binario usual es: ' + str(math.ceil(np.log2(len(distr_es)))*len('fractal')))

print('La decodificacion de 0000111011111111111010 es: ' + decod('0000111011111111111010', tree_en))
print('Codificación de geometria en ingles es: ' + codif('geometria', tree_en))
print('La decodificación de ' + codif('geometria', tree_en) + ' en ingles es: ' + decod(codif('geometria', tree_en), tree_en))

print('El índice de Gini es: ' + str(gini(distr_en)))
print('El índice de diversidad de Hill es: ' + str(hill(distr_en)))

