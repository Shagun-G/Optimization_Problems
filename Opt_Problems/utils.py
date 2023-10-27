import numpy as np
from sklearn.datasets import load_svmlight_file

def datasets_manager(name, location):

        X, y = load_svmlight_file(location)

        # preprocessing for specific datasets

        if name.lower() == "mushroom":
            # the target variable needs to be offset
            y = y - 1
        elif name.lower() == "australian":
            # the target has to be changed from {-1, +1} to {0, 1}
            y[y == -1] = 0
        elif name.lower() == "phishing":
            # no formatting required, {0,1} labels
            pass
        elif name.lower() == "sonar":
            # the target has to be changed from {-1, +1} to {0, 1}
            y[y == -1] = 0
        elif name.lower() == "gisette":
            # the target has to be changed from {-1, +1} to {0, 1}
            y[y == -1] = 0
        elif name.lower() == "a9a":
            # the target has to be changed from {-1, +1} to {0, 1}
            y[y == -1] = 0
        elif name.lower() == "w8a":
            # the target has to be changed from {-1, +1} to {0, 1}
            y[y == -1] = 0
        elif "ijcnn" in name.lower():
            # the target has to be changed from {-1, +1} to {0, 1}
            y[y == -1] = 0
        elif "real-sim" in name.lower():
            # the target has to be changed from {-1, +1} to {0, 1}
            y[y == -1] = 0
        else:
            raise Exception("Unknown dataset, preprocessing might be required for correct format")

        if name not in location:
            raise Exception("Name and file pointed to in location are different")

        return X, y