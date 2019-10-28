import re

class NGramm():
    def train(data):
        normalized_requests = []
        for request in data:
            normalized_requests.append(normalize_request(request['Request']))
        


def normalize_request(request):
    regex = re.compile(r"[a-zA-Z0-9]+")
    replaced = re.sub(regex, '@',request)
    return replaced

# Main method to test if the class works as expected
def main():

    teststring = '../Logfiles/Labeled/normalTrafficTraining.txt'
    expected = '../@/@/@.@'
    result = normalize_request(teststring)
    print(expected == result)
    print(result)

if __name__ == "__main__":
    main()