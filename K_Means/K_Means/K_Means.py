import math

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

