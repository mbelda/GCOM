import numpy as np

from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.datasets import make_blobs

import matplotlib.pyplot as plt

#Creamos los datos
centers = [[1, 1], [-1, -1], [1, -1]]
X, labels_true = make_blobs(n_samples=1000, centers=centers, cluster_std=0.4,
                            random_state=0)

#Clasificacion y coeficiente de Silhouette para el algoritmo Kmeans 
# con k entre 1 y 15
for n_clusters in range(1,16):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    labels = kmeans.labels_
    if n_clusters != 1:
        silhouette = metrics.silhouette_score(X, labels)
        plt.plot(n_clusters, silhouette, '-ob')
    else:
        #Si solo hay un cluster, el coeficiente de Silhouette es -1
        silhouette = -1

plt.xlabel("Nº clusters")
plt.ylabel("Silhouette")
plt.show()



#Clasificacion y coeficiente de Silhouette para el algoritmo DBSCAN
#con la metrica pasada por argumento
def algDBSCAN(metrica):
    f1 = plt.figure()
    ax1 = f1.add_subplot(111)
    plt.xlabel("Epsilon")
    plt.ylabel("Silhouette")
    f2 = plt.figure()
    ax2 = f2.add_subplot(111)
    plt.xlabel("Nº clusters")
    plt.ylabel("Silhouette")
      
    
    #Variamos epsilon en (0.1, 1)
    for epsilon in np.linspace(0.11, 1, num = 40, endpoint=False): 
        #Algoritmo DBSCAN con al menos 10 elementos
        db = DBSCAN(eps=epsilon, min_samples=10, metric=metrica).fit(X)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_
        
        # Numero de clusters en labels, ignorando el ruido.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)
       
        if len(set(labels)) != 1:
            silhouette = metrics.silhouette_score(X, labels)
        else :
            #Si solo hay un cluster, el coeficiente de Silhouette es -1
            silhouette = -1
            
        ax1.plot(epsilon, silhouette, '-ob')
        
        ax2.plot(n_clusters_, silhouette, 'or')
        
    
    
    plt.show()

algDBSCAN('euclidean')
algDBSCAN('manhattan')

    