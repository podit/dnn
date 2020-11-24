'''
    Import class for creating numpy arrays of the MNIST database
        - Requires pip package: python-mnist
        - Reads files from 'data' directory
            * Get data and decompress manually from:
                http://yann.lecun.com/exdb/mnist/
            * Run script which comes with python-mnist:
                - Depending on root or user pip install path:
                    * `/bin/mnist_get_data.sh`
                    * `~/.local/bin/mnist_get_data.sh`
'''
class imp:
    # Import MNIST data using python_mnist package and process into arrays
    def __init__(self):
        from mnist import MNIST
        self.mndata = MNIST('data')

    # Take lists from python_mnist and create numpy arrays of data
    def preProc(self, images, labels, vanishingMean, twoD):
        import numpy as np
        # Create numpy arrays from image data
        imgs = np.array(images, dtype=np.float32)

        # Normalise image data
        if vanishingMean: imgs = (imgs / 127.5)
        else: imgs = imgs / 255
        # Reshape image data to 2D if needed
        if twoD: imgs = imgs[:].reshape((-1, 28))

        # Create one-hot label vectors from label data
        labs = np.eye(10)[labels]

        return imgs, labs

    # Import training data
    def importTrain(self, vanishingMean = False, twoD = False):
        images, labels = self.mndata.load_training()
        imgsTrain, labsTrain = self.preProc(images, labels, vanishingMean, twoD)
        return(imgsTrain, labsTrain)
    # Import testing data
    def importTest(self, vanishingMean = False, twoD = False):
        images, labels = self.mndata.load_testing()
        imgsTest, labsTest = self.preProc(images, labels, vanishingMean, twoD)
        return(imgsTest, labsTest)

