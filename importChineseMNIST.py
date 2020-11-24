import numpy as np
import pandas as pd

class imp:
    # Import class for use with jpg input data and csv label set
    #   Chinese MNIST - https://www.kaggle.com/gpreda/chinese-mnist
    def __init__(self):
        self.mndata = 'data'
        self.mninput = self.mndata + '/data/data'
        self.mnlabel = self.mndata + '/chinese_mnist.csv'
        self.classes = 15
        self.samples = 1000

    def importLabels(self):
        df = pd.read_csv(self.mnlabel)
        return df
