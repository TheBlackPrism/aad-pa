import re
import urllib.parse
# Dictionary Structure Example of a Request
#
# Type: GET
# Request: http://localhost:8080/tienda1/index.jsp HTTP/1.1
# User-Agent: Mozilla/5.0 (compatible; Konqueror/3.5; Linux) KHTML/3.5.8 (like Gecko)
# Pragma: no-cache
# Cache-control: no-cache
# Accept: text/xml,application/xml,application/xhtml+xml,text/html;q=0.9,text/plain;q=0.8,image/png,*/*;q=0.5
# Accept-Encoding : x-gzip, x-deflate, gzip, deflate
# Accept-Charset: utf-8, utf-8;q=0.5, *;q=0.5
# Accept-Language: en
# Host: localhost:8080
# Cookie: JSESSIONID=1F767F17239C9B670A39E9B10C3825F4
# Connection: close

def read_data(url):
    """Reads Data from the given path and returns a list of dictionaries of the requests.
A request dictionaire is looking like the following example
(In case of a GET Request there is no Content-Length and Parameters entry):
    Type: POST
    Request: http://localhost:8080/tienda1/publico/anadir.jsp HTTP/1.1
    User-Agent: Mozilla/5.0 (compatible; Konqueror/3.5; Linux) KHTML/3.5.8 (like Gecko)
    Pragma: no-cache
    Cache-control: no-cache
    Accept: text/xml,application/xml,application/xhtml+xml,text/html;q=0.9,text/plain;q=0.8,image/png,*/*;q=0.5
    Accept-Encoding: x-gzip, x-deflate, gzip, deflate
    Accept-Charset: utf-8, utf-8;q=0.5, *;q=0.5
    Accept-Language: en
    Host: localhost:8080
    Cookie: JSESSIONID=1F767F17239C9B670A39E9B10C3825F4
    Connection: close
    Content-Length: 68
    Parameters: id=3&nombre=Vino+Rioja&precio=100&cantidad=55&B1=A%F1adir+al+carrito
    """
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
            line = urllib.parse.unquote(line, encoding = "utf-8")
            if line == '\n':
                if isreadingblock and not wasemptyline:
                    wasemptyline = True

                elif wasemptyline:
                    isreadingblock = False
                    data.append(__get_dictionary_from_request(request))
                    request = []
            else:
                if wasemptyline and isreadingblock:
                    request.append("Parameters: " + line)

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

    return __remove_eol_from_requests(data)

def __get_dictionary_from_request(request):
    """Returns a Dictionary of the request-string.
    """
    dict = {}

    for pair in request:
        s = pair.split(": ", 1)
        if len(s) == 2:
            dict[s[0]] = s[1]
        else:
            dict['Unknown'] = s[0]

    return dict

def __remove_eol_from_requests(requests):
    """Removes all end of lines from a list of dictionaires
    """
    regex = re.compile(r"\n$")
    list = []
    for request in requests:
        for parameter in request:
            request[parameter] = re.sub(regex, '', request[parameter])

        list.append(request)
    return list

def append_parameter_to_request(requests, ignore_empty_parameters=False):
    """Appends the parameter of a post request at the end of the requested 
url sepparated by a ? and removes HTML/*.* from url.
Requests without parameter are ignored when the flag is set to true.
    """
    regex = re.compile(r"( HTTP/.\..)$")
    list = []
    for entry in requests:
        request = entry["Request"]
        replaced = re.sub(regex, '',request)

        if "Parameters" in entry:
            parameter = "?" + entry.pop("Parameters")
        else:
            parameter = ""

        entry["Request"] = replaced + parameter
        if not ignore_empty_parameters:
            list.append(entry)
        elif not entry["Request"].find("?") == -1:
            list.append(entry)
    return list

def write_csv(filename, ngrams, feature_vectors):
    """Writes Feature vectors and their names into the specified file.
    """
    ngram = list(ngrams.keys())
    first_vector = feature_vectors[0]
    f = open(filename, "w", encoding="utf-8")
    for i in range(len(ngram)):
        if i > 0:
            f.write(",")
        f.write(ngram[i])
    f.write("\n")
    
    for vector in feature_vectors:
        for i in range(len(vector)):
            if i > 0:
                f.write(",")
            f.write(str(vector[i]))
        f.write("\n")
    f.close()

def read_csv(url, unnecessary_columns = 0):
    """Reads a CSV file and returns the feature vectors as an np array
    """
    dict = pd.read_csv(url).to_dict('index')
    ngrams = list(dict.values())
    features = []
    for tuple in ngrams:
        # This is a hack to remove the last elements of the featurevectors and convert the rest to floats
        feature = np.asarray(list(tuple.values()))
        features.append(list(map(float, feature[:-unnecessary_columns]))) 
    return np.asarray(features)

