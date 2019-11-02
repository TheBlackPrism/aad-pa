import re
import K_Means
import logfileparser as parser

class NGramm():

    def __init__(self, n=2):
        self.n = n
        self.ngramms = {}
        self.total_number_ngramms = 0
        self.ngramms_probability = {}

    def fit(self, data):
        self.ngramms = self.get_ngramms_for_all(data)
        self.total_number_ngramms = sum(self.ngramms.values())

        for ngramm in self.ngramms:
            self.ngramms_probability[ngramm] = float(self.ngramms[ngramm]) / self.total_number_ngramms

    def get_probability_of_request(self, data):
        ngramms = get_ngramms_for_request(data)
        total_probability = 0
        for ngramm in ngramms:
            probability_ngramm = self.ngramms_probability.get(ngramm)
            total_probability += probability_ngramm
        return total_probability / len(ngramms)

    def get_ngramms_for_all(self, data):
        ngramms = {}
        normalized_requests = []

        for request in data:
            normalized_requests.append(normalize_request(request['Request']))
        for request in normalized_requests:
            for i in range(len(request)):
                ngramm = request[i:i + self.n] # Split a requests into the n-gramms for the length of n
                if ngramm in ngramms:
                    ngramms[ngramm] += 1
                else:
                    ngramms[ngramm] = 1

        return ngramms

    def get_ngramms_for_request(self, request):
        ngramms = {}
        normalized_request = normalize_request(request['Request'])
        for i in range(len(request)):
            ngramm = request[i:i + self.n]
            if ngramm in self.ngramms:
                ngramms[ngramm] += 1
            else:
                ngramms[ngramm] = 1
        return ngramms

def normalize_request(request):
    regex = re.compile(r"[a-zA-Z0-9]+")
    replaced = re.sub(regex, '@',request)
    regex = re.compile(r"\n")
    replaced = re.sub(regex, '', replaced)
    return replaced

def main():
    #read data here
    training_data = parser.read_data('../Logfiles/Labeled/normalTrafficTraining.txt')

    print("\n**************************")
    print("Training model:")

    #fit data
    ng = NGramm()
    ng.fit(training_data)
    
    print("\n**************************")
    print("All N-Gramms:")
    print(ng.ngramms)
    
    print("\n**************************")
    print("N-Gramms probabilities:")
    print(ng.ngramms_probability)
    
    print("\n**************************")
    print("Total N-Gramms:")
    print(ng.total_number_ngramms)

    
    test_clean = parser.read_data('../Logfiles/Labeled/normalTrafficTest.txt')
    test_anomalous = parser.read_data('../Logfiles/Labeled/anomalousTrafficTest.txt')

    km = K_Means.K_Means()
    km.fit()

if __name__ == "__main__":
    main()