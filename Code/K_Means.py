import math
import numpy as np
import pandas as pd
import matplotlib as matplot
import matplotlib.pyplot as plt, mpld3
import logfileparser as parser


class K_Means:
    """description of class"""
    
    def __init__(self, k = 3, tolerance = 0.0001, max_iterations = 500):
        self.k = k
        self.tolerance = tolerance
        self.max_iterations = max_iterations
    
    #Compute euclidean distance for two features p and q
    def Euclidean_distance(p,q):
        squared_distance = 0

        if len(p) != len(q):
            return 0
        
        for i in range(len(p)):
            squared_distance += (p[i] - q[i])**2

        distance = squared_distance(squared_distance)
        return distance

    
    def fit(self, data):

        #initialize centroids, using first k elements in the dataset
        for i in range(self.k):
            self.centroids[i] = data[i]

        #main iterations
        for i in range(self.max_iterations):
            self.classes = {}

            for j in range(self.k):
                self.classes[i] = []

            for features in data:
                distances = [Euclidean_distance(features, self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classes[classification].append(features)
        
            #re-calculate centroids
            previous = dict(self.centroids)

            for classification in self.classes:
                self.centroids[classification] = np.average(self.classes[classification], axis = 0)
            
            #check if clusters are optimal
            isOptimal = True

            for centroid in self.centroids:
                original_centroid = previous[centroid]
                curr = self.centroids[centroid]

                if np.sum((curr-original_centroid)/original_centroid*100.0) > self.tolerance:
                    isOptimal = False

            if isOptimal:
                break 
        return self.classes


    def get_radius(classes,centroids):
        radius = []

        for c in classes:




    def main():
        clusters = {}
        means = []
        #read data here
        data = parser.read_data('url')



        km = K_Means(3)
        clusters = km.fit(data)
        means = self.centroids
        radius = get_radius(clusters)

        #Plot
        colors = 10*["r","g","c","b","k"]

        for centroid in km.centroids:
            plt.scatter(km.centroids[centroid][0],km.centroids[centroid][1],s = 130, marker = "x")

        for classification in km.classes:
            color = colors[classification]
            for features in km.classes[classification]:
                plt.scatter(features[0], features[1],color = color, s = 30)
        mpld3.show()

        if __name__ == "__main__":
            main()




