# -*- coding: utf-8 -*-
"""
Referencias:
    
    Fuente primaria del reanálisis
    https://www.esrl.noaa.gov/psd/data/gridded/data.ncep.reanalysis2.pressure.html
    
    Altura geopotencial en niveles de presión
    https://www.esrl.noaa.gov/psd/cgi-bin/db_search/DBListFiles.pl?did=59&tid=81620&vid=1498
    
    Temperatura en niveles de presión:
    https://www.esrl.noaa.gov/psd/cgi-bin/db_search/DBListFiles.pl?did=59&tid=81620&vid=1497

"""
from __future__ import division
import os
import datetime as dt  # Python standard library datetime  module
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import netcdf as nc
from sklearn.decomposition import PCA
import math

workpath = r"C:\Users\Majo\Desktop\P4 gcom"
os.getcwd()
#os.chdir(workpath)
files = os.listdir(workpath)

def errorTempMedia(minimos, temps, temp0p, rangeLon, rangeLat, rangeP, p):
    media = np.zeros((rangeLat,rangeLon))
    
    
    # Para cada latitud, longitud sumamos las temperaturas de los 4 días de mínimos en el nivel de presión p
    for i in range(len(minimos)):
        for lon in range(rangeLon):
            for lat in range(rangeLat):
                    media[lat, lon] = media[lat, lon] + temps[minimos[i][0],p,lat,lon]
    
    # Dividimos entre 4 todas esas sumas para obtener la media
    media = media/4
    
    eam = np.average(np.absolute(np.subtract(media, temp0p)))

    return eam
    
def calculaAnalogos(altgeo, altgeo0):
    minimos = [[0,1000000], [0,1000000], [0,1000000], [0,1000000]]
    
    # Calculamos la distancia para cada día
    for t in range(len(time)):
        distancia = 0
        for lat in range(len(lats)):
            for lon in range(len(lons)):
                #dist = 0
                #p = 1000 -> 0
                #p = 500  -> 5
                for p in {0,5}:
                    distancia = distancia + 0.5*(altgeo0[p,lat,lon] - altgeo[t,p,lat,lon])**2
        distancia = math.sqrt(distancia)       
        
        
        # Si la distancia es menor que la mayor que hay en minimos, se sustituye
        if distancia < minimos[0][1] :
            minimos[0]=[t,distancia]
            minimos.sort(key=lambda tup: tup[1], reverse=True)
    
    return minimos

#Cargamos los valores de Z
f = nc.netcdf_file(workpath + "/hgt.2019.nc", 'r')

time = f.variables['time'][:].copy() #t
level = f.variables['level'][:].copy() #p
lats = f.variables['lat'][:].copy()
lons = f.variables['lon'][:].copy()
hgt = f.variables['hgt'][:].copy()
offset = f.variables['hgt'].add_offset
scale = f.variables['hgt'].scale_factor
hgt = scale * hgt + offset

f.close()

"""
Ejemplo de evolución temporal de un elemento de aire
"""

plt.plot(time, hgt[:, 1, 1, 1], c='r')
plt.show()

#time_idx = 237  # some random day in 2012
# Python and the renalaysis are slightly off in time so this fixes that problem
# offset = dt.timedelta(hours=0)
# List of all times in the file as datetime objects
dt_time = [dt.date(1800, 1, 1) + dt.timedelta(hours=t) #- offset\
           for t in time]
np.min(dt_time)
np.max(dt_time)


"""
Distribución espacial de la temperatura en el nivel de 1000hPa, para el primer día
"""
plt.contour(lons, lats, hgt[1,5,:,:])
plt.show()

#Fijamos la presión a 500hPa
hgt2 = hgt[:,5,:,:].reshape(len(time),len(lats)*len(lons))

#Teniendo en cuenta Z, estimamos las 4 comp prals
n_components=4

X = hgt2
Y = hgt2.transpose()
pca = PCA(n_components=n_components)

pca.fit(X)
print("% Varianza explicada: " + str(pca.explained_variance_ratio_))
out = pca.singular_values_

pca.fit(Y)
print("% Varianza explicada: " + str(pca.explained_variance_ratio_))
out = pca.singular_values_

State_pca = pca.fit_transform(X)


Element_pca = pca.fit_transform(Y)
Element_pca = Element_pca.transpose(1,0).reshape(n_components,len(lats),len(lons))

fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.4)
for i in range(1, n_components + 1):
    ax = fig.add_subplot(2, 2, i)
    ax.text(0.5, 90, 'PCA-'+str(i),
           fontsize=18, ha='center')
    plt.contour(lons, lats, Element_pca[i-1,:,:])
plt.show()


#Cargamos a0 y restringimos el sistema
f = nc.netcdf_file(workpath + "/hgt.2020.nc", 'r')
level = f.variables['level'][:].copy() #p
lats = f.variables['lat'][:].copy()
lons = f.variables['lon'][:].copy()
hgt0 = f.variables['hgt'][:].copy()
offset = f.variables['hgt'].add_offset
scale = f.variables['hgt'].scale_factor
hgt0 = scale * hgt0 + offset


lons = lons - 180

lons = lons[65:80] # Entre 65 y 79 están las longitudes en (-20º, 20º) (intervalo abierto)
lats = lats[17:24] # Entre 17 y 23 están las latitudes en (30º, 50º) (intervalo abierto)
hgt0 = hgt0[19,:,17:24,65:80] 
hgt = hgt[:,:,17:24,65:80]

#Calculamos los 4 días más análogos
analogos = calculaAnalogos(hgt, hgt0)
analogos.sort(key=lambda x: x[0])
print("Los cuatro días más análogos son:")
for i in range(4):
    print(dt.datetime(2019,1,1) + dt.timedelta(analogos[i][0]))

#Cargamos las temperaturas para predecir la del dia a0
f = nc.netcdf_file(workpath + "/air.2019.nc", 'r')
air = f.variables['air'][:].copy()
offset = f.variables['air'].add_offset
scale = f.variables['air'].scale_factor
air = scale * air + offset

#Cargamos la temperatura del día a0
f = nc.netcdf_file(workpath + "/air.2020.nc", 'r')
air0 = f.variables['air'][:].copy()
offset = f.variables['air'].add_offset
scale = f.variables['air'].scale_factor
air0 = scale * air0 + offset

#Las restringimos
air0 = air0[19,0,:,:]

print("El error en la temperatura predicha para el día a0 es: " + \
      str(errorTempMedia(analogos, air, air0, 144, 73, len(level), 0)))