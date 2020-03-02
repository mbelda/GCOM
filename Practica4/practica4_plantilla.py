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


#f = nc.netcdf_file(workpath + "/" + files[0], 'r')
f = nc.netcdf_file(workpath + "/hgt.2019.nc", 'r')

print("Hist " + str(f.history))
print("Dim " + str(f.dimensions))
print("Var " + str(f.variables))
time = f.variables['time'][:].copy() #t
time_bnds = f.variables['time_bnds'][:].copy() #
time_units = f.variables['time'].units
level = f.variables['level'][:].copy() #p
print("Level " + str(level))
lats = f.variables['lat'][:].copy()
print("lats " + str(lats))
lons = f.variables['lon'][:].copy()
print("lons " + str(lons))
hgt = f.variables['hgt'][:].copy()
print(hgt.shape)

f.close()

"""
Ejemplo de evolución temporal de un elemento de aire
"""
#air[t,p,y,x]
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

hgt2 = hgt[:,5,:,:].reshape(len(time),len(lats)*len(lons))
#air3 = air2.reshape(len(time),len(lats),len(lons))

n_components=4


X = hgt2
Y = hgt2.transpose()
pca = PCA(n_components=n_components)

pca.fit(X)
print(pca.explained_variance_ratio_)
out = pca.singular_values_

pca.fit(Y)
print(pca.explained_variance_ratio_)
out = pca.singular_values_


State_pca = pca.fit_transform(X)
#Ejercicio de la práctica
Element_pca = pca.fit_transform(Y)
Element_pca = Element_pca.transpose(1,0).reshape(n_components,len(lats),len(lons))

fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.4)
for i in range(1, 5):
    ax = fig.add_subplot(2, 2, i)
    ax.text(0.5, 90, 'PCA-'+str(i),
           fontsize=18, ha='center')
    plt.contour(lons, lats, Element_pca[i-1,:,:])
plt.show()

f = nc.netcdf_file(workpath + "/air.2020.nc", 'r')

level = f.variables['level'][:].copy() #p
lats = f.variables['lat'][:].copy()
lons = f.variables['lon'][:].copy()
air = f.variables['air'][:].copy()

Ta0 = air[19,0,0,0]
xa0 = lats[0]
ya0 = lons[0]
pa0 = level[0]

f = nc.netcdf_file(workpath + "/air.2019.nc", 'r')

print("Hist " + str(f.history))
print("Dim " + str(f.dimensions))
print("Var " + str(f.variables))
time = f.variables['time'][:].copy() #t
time_bnds = f.variables['time_bnds'][:].copy() #
time_units = f.variables['time'].units
level = f.variables['level'][:].copy() #p
lats = f.variables['lat'][:].copy()
lons = f.variables['lon'][:].copy()
air = f.variables['air'][:].copy()
air_units = f.variables['air'].units
print(air.shape)

#for i in range(len(lons)):
#    if lons[i] == 30 or lons[i] == 50:
#        print(i)

lats = lats[28:44]
lons = lons[12:20]

def dist_eucla0(p,x,y ,T):
    if level[p] == 500 or level[p] == 1000:
        lp = 0.5
    else:
        lp = 0
    return math.sqrt((xa0 - lats[x])**2 + (ya0 - lons[y])**2 + (lp*(pa0 - level[p]))**2 + (Ta0 - T)**2)

minimos = [[0,1000]]
for t in range(len(time)):
    for lat in range(len(lats)):
        for lon in range(len(lons)):
            for p in range(len(level)):
                dist = dist_eucla0(p, lat, lon, air[t,p,lat,lon])
                if dist < minimos[0][1]:
                    minimos[0] = [t, dist]
         
print(minimos)