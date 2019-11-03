import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import logfileparser as parser


class K_Means_2(object):
    """description of class"""
    k = 2
    training_data = parser.read_data('../Logfiles/Labeled/normalTrafficTraining.txt')

    #Test feature could be length of request.
    #Parse data, extract requests and put their lengths in a list
    requests = ["" for x in range(len(training_data))]
    request_lengths = np.zeros([len(training_data)])
    for i in range(len(training_data)):
        tmp = training_data[i]
        requests[i] = tmp.get('Request', None)
        req = requests[i]
        request_lengths[i] = len(req)

    print("\n**************************")
    print("Training model:")
    print("k = %d" %k)

    #fit trainings data to obtain clusters
    kmeans = KMeans(k)
    kmeans.fit(request_lengths)

    clusters = kmeans.labels_.tolist()

    for i in range(len(request_lengths)):
        if cluster[i] == 1:
            cluster1 = plt.scatter(request_lengths[i], c = 'r', marker = 'x')
        elif clusters[i] == 0:
            cluster2 = plt.scatter(request_lengths[i], c = 'g', marker = 'o')
            


    plt.plot(clusters)

    #test clean data
    print("**************************")
    print("Testing normal traffic:")
    test_data = parser.read_data('../Logfiles/Labeled/normalTrafficTest.txt')
    test_requests = []
    test_request_lengths = []
    malicious_requests = []

    test_requests = ["" for x in range(len(training_data))]
    test_request_lengths = np.zeros([len(training_data)])
    for i in range(len(test_data)):
        tmp = test_data[i]
        test_requests[i] = tmp.get('Request', None) #returns None if key doesn't exist
        test_req = test_requests[i]
        test_request_lengths[i] = len(test_req)
  
    kmeans.predict(test_request_lengths)





