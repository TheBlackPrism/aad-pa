import numpy as np

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

    def build_feature_vector(self, url_lengths):
        feature_vector = []
        

        for i in range(len(url_lengths)):
            lengths = []
            lengths.append(url_lengths[i])
            feature_vector.append(lengths.copy())
            lengths.clear()

        return np.asarray(feature_vector)

    def extract_feature(self,data):
        data_urls = self.get_urls(data)
        url_lengths = self.get_url_lengths(data_urls)
        feature_vector = self.build_feature_vector(url_lengths)

        return feature_vector
