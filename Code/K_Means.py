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

        clusters = {}

        #initialize centroids, using first k elements in the dataset
        for i in range(self.k):
            self.centroids[i] = data[i]

        #main iterations
        for i in range(self.max_iterations):
            self.classes = {}

            for j in range(self.k):
                self.classes[j] = []
            
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
                
                #clusters = dict(self.classes)
                #for classification in self.classes:
                    #clusters[classification] = self.centroids[classification]
                break 
        return centroids


    def get_radius(classes,centroids,attr):
        radius = {}

        for cent in centroids:
            max_Dist = Euclidean_distance(cent,attr)
            radius[cent] = ((np.absolute(classellipsis[cent] + 1)/np.absolute(classes[cent])) * max_Dist
        return radius
        




    def main():
        clusters = {}
        means = []
        requests = []
        request_lengths = []
        #read data here
        training_data = parser.read_data('url')

        #Test feature could be length of request.
        #Parse data, extract requests and put their lengths in a list
        for i in range(len(training_data)):
            tmp = training_data[i]
            requests[i] = tmp.get('Request', None)
            req = requests[i]
            request_lengths[i] = len(req)
        
        




        #fit data
        km = K_Means(3)
        means = km.fit(request_lengths)
        radius = get_radius(km.classes,means,'attr')

        #test new data
        test_data = parser.read_data('url')
        test_requests = []
        test_request_lengths = []
        dist = 0
        counter = 0
        malicious_requests = []


        for i in range(len(test_data)):
            tmp = test_data[i]
            test_requests[i] = tmp.get('Request', None) #returns None if key doesn't exist
            test_req = test_requests[i]
            test_request_lengths[i] = len(test_req)

        
        for i in range(len(test_request_lengths)):
            for m,r in radius:
                dist = Euclidean_distance(test_request_lengths[i],m)
                if dist > r:
                    malicious_requests[counter] = test_request_lengths[i]
                    counter += 1


        if len(malicious_requests) > 0:
            print("No malicious requests detected.")
        else:
            for i in range(len(malicious_requests)):
                print(malicious_requests[i])

            
    

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




