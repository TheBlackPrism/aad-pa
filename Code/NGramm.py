import re
import logfileparser as parser

class NGramm():

    def __init__(self, n=2):
        self.n = n
        self.ngramms = {}
        self.total_ngramms = 0
        self.ngramms_probability = {}

    def fit(self, data):

        self.get_ngramms = get_ngramms(data)
        self.total_ngramms = sum(self.ngramms.values())

        for ngramm in self.ngramms:
            self.ngramms_probability[ngramm] = float(self.ngramms[ngramm]) / self.total_ngramms

    def get_probability(self, data):
        request

    def get_ngramms(self, data):
        ngramms = {}
        normalized_requests = []

        for request in data:
            normalized_requests.append(normalize_request(request['Request']))

        for request in normalized_requests:
            for i in range(len(request)):
                ngramm = request[i:i + self.n]
                if ngramm in self.ngramms:
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
    ngramms = ng.fit(training_data)
    print(ng.ngramms)
    print(ng.ngramms_probability)
    print(ng.total_ngramms)


if __name__ == "__main__":
    main()