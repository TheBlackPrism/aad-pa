import math
import numpy as np

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

    
    def main():
        if __name__ == "__main__":
            main()

        for i in range(self.k):
            self.centroids[i] = data[i]

        for i in range(self.max_iterations):
            self.classes = {}

            for j in range(self.k):
                self.classes[i] = []

            for features in data:
                distances = [Euclidean_distance(features, self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classes[classification].append(features)
        
            previous = dict(self.centroids)

            for classification in self.classes:
                self.centroids[classification] = np.average(self.classes[classification], axis = 0)
            
            isOptimal = True

            for centroid in self.centroids:
                original_centroid = previous[centroid]
                curr = self.centroids[centroid]

                if np.sum((curr-original_centroid)/original_centroid*100.0) > self.tolerance:
                    isOptimal = False

            if isOptimal:
                break 




