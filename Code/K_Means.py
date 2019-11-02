import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans


class K_Means_2(object):
    """description of class"""

    data = np.array([])
    plt.scatter(data[:,0], data[:,1], label='True Position')

    kmeans = KMeans(n_clusters = 2)
    kmeans.fit(data)


