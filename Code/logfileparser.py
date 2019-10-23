def read_data(url):
    data = []
    request = []
    file = open(url, "r")

    if file.mode == 'r':
        lines = file.readlines()

        isreadingblock = False
        wasemptyline = True
        
        # Attention! line is never validated
        # Whole parser is rather fragile
        for line in lines:
            if line == '\n':
                if isreadingblock and not wasemptyline:
                    wasemptyline = True

                elif wasemptyline:
                    isreadingblock = False
                    data.append(__get_dictionary_from_request(request))
                    request = []
            else:
                if wasemptyline and isreadingblock:
                    request.append("Response: " + line)

                elif wasemptyline:
                    isreadingblock = True
                    wasemptyline = False
                    args = line.split(" ", 1)

                    if len(args) == 2:
                        request.append("Type: " + args[0])
                        request.append("Request: " + args[1])

                elif not isreadingblock:
                    isreadingblock = True
                    wasemptyline = False
                    request.append(line)

                else:
                    request.append(line)
    else:
        print("File could not be opened")

    return data

def __get_dictionary_from_request(request):
    dict = {}

    for pair in request:
        s = pair.split(": ", 1)
        if len(s) == 2:
            dict[s[0]] = s[1]
        else:
            dict['Unknown'] = s[0]

    return dict

if __name__ == '__main__':

    # Test by reading all the datasets
    dict = read_data("../Logfiles/Labeled/normalTrafficTraining.txt")
    dict = read_data("../Logfiles/Labeled/normalTrafficTest.txt")
    dict = read_data("../Logfiles/Labeled/anomalousTrafficTest.txt")
