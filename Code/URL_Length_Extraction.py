class URL_Length_Extraction(object):
    """description of class"""

    def __init__(self):
        pass

    def get_urls(self, parsed_dataset):
        url_list = []

        for i in range(len(parsed_dataset)):
            url = parsed_dataset[i].get('Request')
            url_list.append(url)

        return url_list

    def get_url_lengths(self, url_list):
        url_lengths = []

        for i in range(len(url_list)):
            length = len(url_list[i])
            url_lengths.append(length)

        return url_lengths


        





               




def main():


    print("**************************")
    print("Reading data...")

    # Reading Data
    training_data = parser.read_data('../Logfiles/Labeled/normalTrafficTraining.txt')
    test_clean = parser.read_data('../Logfiles/Labeled/normalTrafficTest.txt')
    test_anomalous = parser.read_data('../Logfiles/Labeled/anomalousTrafficTest.txt')

    training_data = parser.append_parameter_to_request(training_data)
    test_clean = parser.append_parameter_to_request(test_clean)
    test_anomalous = parser.append_parameter_to_request(test_anomalous)
    print("**************************")
    print("Extracting URL Length...")
    
    urlLength = URL_Length_Extraction()
    url_list = urlLength.get_urls(training_data)
    url_length_list = urlLength.get_url_lengths(url_list)
    
   

   
   
    
    print("Done.")
    print("**************************")


if __name__ == "__main__":
    main()








