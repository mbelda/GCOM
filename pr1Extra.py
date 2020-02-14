import numpy as np 
import imageio as imgio
from PIL import Image 


iteracionesTotales = 7
#Tamaño total de la imagen
tam = 3**iteracionesTotales
#Creamos el cuadrado grande y lo pintamos negro
cuadrado = np.empty([tam, tam, 3], dtype = np.uint8)
cuadrado.fill(0)

blanco = np.array([255, 255, 255], dtype = np.uint8) 
 
for i in range(iteracionesTotales + 1): 
    tamCuadradoi = 3**(iteracionesTotales - i)
    #Para cada iteración busco la medida de hausdorff
    if i != 0:
        Nr = 8**i
        r = (1/3)**i
        medida = np.abs(np.log(Nr)/np.log(r))
        print('Medida it', i, ': ', medida)
        
    else:
        print('Log8/log3 =', np.log(8)/np.log(3))
        
    for x in range(3**i): 
        #Busco el cuadrado central
        if x % 3 == 1: 
            for y in range(3**i):
                if y % 3 == 1: 
                    #Lo pinto blanco
                    cuadrado[x * tamCuadradoi : (x + 1) * tamCuadradoi,\
                    y * tamCuadradoi : (y + 1) * tamCuadradoi] = blanco

#Guardar y abrir la imagen
imgio.imwrite('sierpinski.jpg', cuadrado[:,:,0])
im = Image.open("sierpinski.jpg") 
im.show()
