'''
This script performs the basic process for applying a machine learning
algorithm to a dataset using Python libraries.

The four steps are:
   1. Download a dataset (using pandas)
   2. Process the numeric data (using numpy)
   3. Train and evaluate learners (using scikit-learn)
   4. Plot and compare results (using matplotlib)

'''

URL = "../Logfiles/Labeled/normalTrafficTraining.txt"
#URL = "../Logfiles/Unlabeled/access.log01.txt"

import pandas as pa
import numpy as np
import matplotlib.pyplot as plt

try:
    # [OPTIONAL] Seaborn makes plots nicer
    import seaborn
except ImportError:
    pass

# =====================================================================

def download_data():
    frame = pa.read_csv(
        URL,

        # Specify the file encoding
        encoding='utf-8',

        # Specify the separator in the data
        sep=',',            # comma separated values
        #sep='\t',          # tab separated values
        #sep=' ',           # space separated values

        # Ignore spaces after the separator
        skipinitialspace=True,
        # Generate row labels from each row number
        #index_col=0,
        #index_col=0,       # use the first column as row labels
        #index_col=-1,      # use the last column as row labels
    
        # Generate column headers row from each column number
        header=None,
        #header=0,          # use the first line as headers

        # Use manual headers and skip the first row in the file
        #header=0,
        names=['attributes', 'values']
    )
    

    # Return a subset of the columns
    #return frame[['col1', 'col4', ...]]
    
    print(frame.head())
    
    # Return the entire frame
    return frame

# =====================================================================

def read_data():
    data = []
    request = []
    file = open(URL, "r")

    if file.mode == 'r':
        lines = file.readlines()

        isreadingblock = False
        wasemptyline = True
        
        # line is never validated
        for line in lines:
            if line == '\n':
                if isreadingblock and not wasemptyline:
                    wasemptyline = True

                elif wasemptyline:
                    isreadingblock = False
                    data.append(get_dictionary_from_request(request))
                    request = []
            else:
                if wasemptyline and isreadingblock:
                    request.append("Response: " + line)

                elif wasemptyline:
                    isreadingblock = True
                    wasemptyline = False
                    args = line.split(" ", 1)
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

def get_dictionary_from_request(request):
    dict = {}

    for pair in request:
        s = pair.split(": ", 1)
        print(s)
        dict[s[0]] = s[1]

    return dict

if __name__ == '__main__':
    # Loading the data set from URL
    print("Loading data from {}".format(URL))
    #frame = download_data()
    dict = read_data()
    # Process data into feature and label arrays
    #print("Processing {} samples with {} attributes".format(len(frame.index), len(frame.columns)))

