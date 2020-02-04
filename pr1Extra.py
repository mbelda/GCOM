import numpy as np 
import imageio as imgio
from PIL import Image 


iteracionesTotales = 7
#Tamaño total de la imagen
tam = 3**iteracionesTotales
#Creamos el cuadrado grande vacío y lo pintamos blanco
square = np.empty([tam, tam, 3], dtype = np.uint8)
square.fill(0)

negro = np.array([255, 255, 255], dtype = np.uint8) 
 
for i in range(0, iteracionesTotales + 1): 
	tamCuadradoi = 3**(iteracionesTotales - i) 
	for x in range(0, 3**i): 
		#Busco el cuadrado central
		if x % 3 == 1: 
			for y in range(0, 3**i): 
				if y % 3 == 1: 
                    #Lo pinto negro
					square[x * tamCuadradoi : (x + 1) * tamCuadradoi,\
                            y * tamCuadradoi : (y + 1) * tamCuadradoi] = negro 

#Guardar y abrir la imagen
imgio.imwrite('sierpinski.jpg', square[:,:,0])
im = Image.open("sierpinski.jpg") 
im.show()
