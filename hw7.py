#Brian West 3/30/25
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

data = pd.read_csv("Data/Spotify_Youtube.csv")#reads in the data from a file
liveness = data["Liveness"]
energy = data["Energy"]
loudness = data["Loudness"]
x = []#list to hold the data point locations
for i in range(len(liveness)):
    x.append([liveness[i],energy[i],loudness[i]])#only gets the data that we want to use

num_C = 10 #number of clusters
num_inits = 10 #number of times k-means is ran
num_max_iter = 300 #number of iterations on a single k-means run

inertias = []
for i in range (1,num_C):
    km = KMeans(n_clusters = i, n_init = num_inits, max_iter = num_max_iter)
    km.fit(x)
    inertias.append(km.inertia_)

plt.plot(range(1,num_C),inertias, marker = 'o')
plt.title('Elbow Graph')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()
#print((y_km == target).sum()/len(y_km))


num_C = 4
km = KMeans(n_clusters = num_C, n_init = num_inits, max_iter = num_max_iter)
y_km = km.fit_predict(x) #returns a list of what data point belongs in what cluster
c_centers = km.cluster_centers_#a list of cluster centers

"""used to visulize"""
k_clusters = {}
for i in range(num_C):#creats a dictionary where each key is a number of clusters
    k_clusters[str(i)] = [[],[],[]]
# print(k_clusters)
for i in range(num_C):#loops over the number of clusters
    for j in range(len(y_km)):#loops over every data point
        if(y_km[j] == i):#gets every point for each cluster
            n_x,n_y,n_z = x[j]
            lists = k_clusters[str(i)]
            lists[0].append(n_x)
            lists[1].append(n_y)
            lists[2].append(n_z)
#print(k_clusters)
#graph the data so we can understand what is going on.
#figure, ax = plt.subplots(nrows=1, ncols=2)#creates 2 subplots
ax = plt.axes(projection='3d')

for i in k_clusters:#plots each cluster
    x,y,z = k_clusters[i]
    ax.scatter(x,y,z, label = str(i))
center = 0
for i in c_centers:#plots the center of the clusters
    x,y,z = i
    ax.scatter(x,y,z, marker = "*", s = 200, label='center ' + str(center))
    center = center + 1
ax.set_title("K-Means")
ax.set_xlabel("Liveness")
ax.set_ylabel("Energy")
ax.set_zlabel("Loudness")
ax.legend()
plt.show()



wardLiveness = linkage(np.reshape(liveness,(len(liveness),1)),method='ward')
wardEnergy = linkage(np.reshape(energy,(len(energy),1)),method='ward')
wardLoudness = linkage(np.reshape(loudness,(len(loudness),1)),method='ward')

dendrogram(wardLiveness)
plt.title('Hierarchical Clustering Liveness')
plt.xlabel('Liveness')
plt.ylabel('Distance')
plt.show()

dendrogram(wardEnergy)
plt.title('Hierarchical Clustering Energy')
plt.xlabel('Energy')
plt.ylabel('Distance')
plt.show()

dendrogram(wardLoudness)
plt.title('Hierarchical Clustering Loudness')
plt.xlabel('Loudness')
plt.ylabel('Distance')
plt.show()
