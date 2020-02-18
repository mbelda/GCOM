"""
Práctica 2
"""

import os
import numpy as np
import pandas as pd
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
distr_en = distr_en.sort_values(by='probab', ascending=True)
distr_en.index=np.arange(0,len(tab_en_states))

tab_es_states = np.array(list(tab_es))
tab_es_weights = np.array(list(tab_es.values()))
tab_es_probab = tab_es_weights/float(np.sum(tab_es_weights))
distr_es = pd.DataFrame({'states': tab_es_states, 'probab': tab_es_probab })
distr_es = distr_es.sort_values(by='probab', ascending=True)
distr_es.index=np.arange(0,len(tab_es_states))


## Ahora definimos una función que haga exáctamente lo mismo
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
 

tree_en = huffman_tree(distr_en)
#print(tree_en)
tree_es = huffman_tree(distr_es)
#print(tree_es)

def codifletra(letra, tree):
    codigo = ''
    for nodo in np.flip(tree):
        keys = np.array(list(nodo.keys()))
        values = np.array(list(nodo.values()))

        if letra in keys[0]:
            codigo = codigo + str(values[0])
        elif letra in keys[1]:
            codigo = codigo + str(values[1])
        
    return codigo


def codif(palabra, tree):
    codigo = ''
    for letra in list(palabra):
        codigo = codigo + codifletra(letra, tree)
        
    return codigo
        
def decod(codigo, tree):
    palabra = ''
    count = 0
    for i in codigo:
        nodo = np.flip(tree)[count]
        keys = np.array(list(nodo.keys()))
       
        if len(keys[int(i)]) == 1:
            palabra = palabra + keys[int(i)]
            count = 0
        else:
            count = buscaSig(keys[int(i)], count, tree)
    return palabra
        
def buscaSig(key, count, tree):
    for i in range(count+1, len(tree)):
        nodo = np.flip(tree)[i]
        keys = np.array(list(nodo.keys()))
        if keys[0][0] in key:
            return i
        
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


def gini(distr):
    GI = 0
    for i in range(len(distr)):
        GI = GI + distr['probab'][i]/len(distr)
    return 1 - 2*GI


def hill(distr):
    hill = 0
    for i in range(len(distr)):
        hill = hill + distr['probab'][i]**2      
    return 1/hill

#Buscar cada estado dentro de cada uno de los dos items
#tree[0].items()[0][0] ## Esto proporciona un '0'
#tree[0].items()[1][0] ## Esto proporciona un '1'
print(decod(codif('español', tree_es), tree_es))
print(decod(codif('fractal', tree_en), tree_en))

print('Codificacion de fractal en ingles: ' + codif('fractal', tree_en))
print('con longitud: ' + str(len(codif('fractal', tree_en))))
print('Codificacion de fractal en español: ' + codif('fractal', tree_es))
print('con longitud: ' + str(len(codif('fractal', tree_es))))
#Para cada caracter hacen falta x bits en funcion del  numero de estados que haya

print('Arbol huffman ingles:')
print(huffmanS(tree_en, distr_en))

print('Arbol huffman español:')
print(huffmanS(tree_es, distr_es))

print(decod('0000111011111111111010', tree_en))

print(gini(distr_en))
print(hill(distr_en))

plt.show()