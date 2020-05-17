'''
    Deep feed-forward neural network implemented using Numpy - Python 3.8
    
        - Performs forward and backward propogation through an abritrary
            layer architecture formed from the input parameters
        - Allows each layer to use an individual activation function:
            - sigmoid, tanh, ReLU, ELU, Softplus, Softmax
        - Built for use with the multiprocessing function
'''

import numpy as np
import numpy.random as npr

class DNN (object):
    # Initialise neural network parameters
    def __init__(self, hidNod, inpNod, outNod, lrnRat, batchSize):
        # Set learning rate
        self.lrnR = lrnRat
        # Set batch size
        self.btc = batchSize
        
        # TODO: Consider integration into class initialisation inputs
        # Set alpha parameter for use in ELU activation function
        self.alpha = .1
        # Set epsilon for padding logarithms from 0 and 1
        self.epsilon = 1e-10
        # Set for biases to be used or not
        self.biases = True

        # Create list of all nodes in network at each layer
        self.nod = [inpNod] + hidNod + [outNod]
        # Set number of input nodes
        self.inpN = inpNod
        # Set hidden layers and dimensions and calculate total hidden nodes
        self.hidLrs = len(hidNod)    
        self.hidN = hidNod
        # Set output nodes and calculate starting index
        self.outN = outNod
        # Calculate total nodes, layers and connected layers
        self.totN = self.inpN + sum(self.hidN) + self.outN
        self.totL = (self.hidLrs - 1) + 2
        self.totV = self.hidLrs + 2
        
        # Create list for weights, biases and deltas
        self.wei = [None] * self.totL
        self.bis = [None] * self.totL
        self.dummyBis = [None] * self.totL
        self.dW = [None] * self.totL
        self.dB = [None] * self.totL
        
        # Iterate through connected layers creating training arrays
        for l in range(self.totL):
            i = l + 1
            self.wei[l] = npr.uniform(-1,1,(self.nod[i],self.nod[l]))
            self.bis[l] = npr.uniform(-1,1,self.nod[i])
            self.dummyBis[l] = np.zeros(self.nod[i])
            # Create delta arrays corresponding to the batch size for averaging
            self.dW[l] = np.zeros((self.nod[i], self.btc))
            self.dB[l] = np.zeros((self.nod[i], self.btc))

        # Display the initialised network
        self.dim()
    
    def reinitialise(self):
        for l in range(self.totL):
            i = l + 1
            self.wei[l] = npr.uniform(-1,1,(self.nod[i],self.nod[l]))
            self.bis[l] = npr.uniform(-1,1,self.nod[i])

    def setActivation(self, hidActs, outAct):
        # Set activation functions to be used in hidden and output layers
        self.hidActs = hidActs
        self.outAct = outAct

    # Function to report dimensions of NN after initialisation
    def dim(self):
        print('Hidden Layers =', self.hidLrs)
        print('Nodes =', self.nod, '\n\tTotal Nodes =', self.totN)
        print('Weights =', self.weights(),
                '\n\tTotal Weights =', sum([w[0]*w[1] for w in self.weights()]))
    # Gets the shape of each layer of weights for `dim` report function
    def weights(self):
        dims = [None] * self.totL
        i = 0
        for l in self.wei:
            dims[i] = l.shape
            i += 1
        return dims

    # Function to save model
    def modelS(self):
        # Create template file names
        wfname = 'modelDNN/wei{:d}'
        bfname = 'modelDNN/bis{:d}'
        # Iterate through weights list and save arrays
        for w in range(len(self.wei)): np.save(wfname.format(w), self.wei[w])
        # Iterate through biases list and save arrays
        if self.biases:
            for b in range(len(self.bis)):
                np.save(bfname.format(b), self.bis[b])
   
    # Function to load model from specified directory
    def modelL(self, modelDir):
        # Create template file names
        wfname = modelDir + '/wei{:d}.npy'
        bfname = modelDir + '/bis{:d}.npy'
        # Iterate through weights list and load arrays
        for w in range(len(self.wei)): self.wei[w] = np.load(wfname.format(w))
        # Iterate through biases list and load arrays
        if self.biases:
            for b in range(len(self.bis)):
                self.bis[w] = np.load(bfname.format(b))
    
    # Function to import input data into the NN
    def importDat(self, img, lab, val, exp):
        val[0] = img
        exp = lab
        return val, exp

    # Activation functions (pass prime as True for derivative)
    def sigmoid(self, x, prime=False): 
        if prime:
            sig = self.sigmoid(x)
            return sig * (1 - sig)
        else: return 1 / (1 + np.exp(-x))

    def tanh(self, x, prime=False):
        if prime: return 1 - self.tanh(x)**2
        else: return 1.7159 * np.tanh((2/3) * x)

    # TODO: np.where functions are backwards, see crossEntropy func
    def relu(self, x, prime=False):
        if prime:
            x = np.where(x < 0, x, 1)
            x = np.where(x == 1, x, 0)
            return x
        else:
            x = np.where(x >= 0, x, 0)
            return x
    def elu(self, x, prime=False):
        if prime:
            x = np.where(x < 0, x, 1)
            x = np.where(x == 1, x, self.alpha * np.exp(x))
            return x
        else:
            x = np.where(x >= 0, x, self.alpha * (np.exp(x) - 1))
            return x

    def softplus(self, x, prime=False):
        if prime: return 1 / (1 + np.exp(-x))
        else: return np.log(1 + np.exp(x))

    def softmax(self, x, prime=False):
        if prime:
            xm = x.reshape((-1,1))
            return np.diagflat(x) - np.dot(xm, xm.T)
        else:
            x -= np.max(x)
            return (np.exp(x) / np.sum(np.exp(x), axis=0)).T
    
    # TODO: Go over confusion about loss functions and revise these
    # Calculate hinge loss
    def hingeLoss(self, yHat, y):
        return np.maximum(0, 1 - yHat * y)
    # Calculate binary cross entropy
    def binaryCrossEntropy(self, yHat, y):
        #err = np.zeros(y.size)
        #err = np.where(y == 1, -np.log(yHat), yHat)
        #err = np.where(y == 0, -np.log(1 - yHat), err)
        # Catrgorical?: np.where y == 1, -np.log(yHat), =np.log(1 - yHat))
        yHat = np.clip(yHat, self.epsilon, 1. - self.epsilon)
        bce = -np.sum(np.sum(y * np.log(yHat + 1e-5)))/self.totN
        return bce
    # Root mean squared error
    def rmse(self, yHat, y):
        e = yHat - y
        rmse = np.sqrt(np.mean(np.square(e)))
        return rmse
    # Mean absolute error
    def mae(self, yHat, y):
        e = yHat - y
        ae = np.absolute(e)
        mae = ae.mean()
        return mae

    # Loss functions used for backpropogation error calculation
    def loss(self, yHat, y):
        return np.subtract(yHat, y)
    def logLoss(self, yHat, y):
        return np.multiply(y, np.log(yHat))

    # Process the input through the NN
    def feedForward(self, inp, val):
        # Define output layer for cleaner indexing
        o = self.totL - 1

        # Iterate through hidden Layers
        for h in range(self.hidLrs):
            # Calculate weighted inputs for layer connections
            weightedInputs = np.multiply(val[h], self.wei[h])
            # Get sum of weighted inputs for each node and add node bias
            weightedSum = np.add(np.sum(weightedInputs, axis=1),
                    self.bis[h] if self.biases else self.dummyBis[h])
            # Record weighted sum for backpropogation step
            inp[h] = weightedSum
            
            # Set node value using the activation function and weighted sum
            val[h+1] = self.hidActs[h](weightedSum)

        # Output Layer calculation (same process as hidden layers)
        weightedInputs = np.multiply(val[o], self.wei[o])
        weightedSum = np.add(np.sum(weightedInputs, axis=1),
            self.bis[o] if self.biases else self.dummyBis[o])
        inp[o] = weightedSum
        
        val[self.totL] = self.outAct(weightedSum)
        
        # Return weighted inputs and node weights for use in backpropogation
        return inp, val

    def backpropogate(self, inp, val, exp, dW, dB):
        # Define output layer index
        o = self.totL - 1
        
        #errorGrad = self.hingeLoss_(self.val[self.totL], self.exp)
        loss = self.logLoss(val[self.totL], exp)
        
        # Calculate output delta using the cost (error) and derivative of output
        #   layer activation function applied to the weighted sum for the layer
        # - If output layer activation is softmax the cost must be multiplied by
        #   the derivative matrix and summed to the error gradient vector
        if self.outAct == self.softmax: delta = np.sum(np.multiply(loss,
            self.outAct(val[self.totL], True)), axis=1)
        else: delta = np.multiply(cost, self.outAct(inp[o], True))
        # Record deltas for weights and biases of the output layer
        dW[o] = delta
        if self.biases: dB[o] = delta
       
       # Backpropogate the delta for the output layers through the weighted
       #    connections of each hidden layer
        for h in reversed(range(self.hidLrs)):
            # Calculate layer delta by multiplying the delta of the previous
            #   layer through the connected weights
            delta = np.multiply(np.dot(delta, self.wei[h+1]),
                    self.hidActs[h](inp[h], True))
            # Record deltas for weights and biases through each layer
            dW[h] = delta
            if self.biases: dB[h] = delta
        
        # Calculate error across all output nodes for profiling training
        error = np.sqrt(np.mean(np.square(loss)))

        # Return deltas for use in batch update (return arrays for re-use)
        return dW, dB, error, inp, val, exp

    # Function to update weights and biases upon batch completeion
    def update(self):
        o = self.totL - 1
        self.btcCount = 0
        # Update weights and biases for hidden layers using deltas
        for h in range(self.hidLrs):
            self.wei[h] = np.transpose(self.wei[h].T -
                    (self.lrnR * np.mean(self.dW[h], 1)))
            if self.biases: self.bis[h] -= self.lrnR * np.mean(self.dB[h], 1)
        # Update weights and biases for output layer using deltas
        self.wei[o] = np.transpose(self.wei[o].T -
                (self.lrnR * np.mean(self.dW[o], 1)))
        if self.biases: self.bis[o] -= self.lrnR * np.mean(self.dB[o], 1)
    
    # Function to control training (optimised for multiprocessing worker pools)
    def run(self, threadNum, threadSize, im, lab):
        # Create numpy array to store errors
        eThr = np.zeros(threadSize)
        dWThr = [None] * self.totL
        dBThr = [None] * self.totL
        # Calculate starting batch index
        start = threadNum-1 * threadSize

        # Create list for node values and initialise the input layer array
        val = [None] * self.totV
        val[0] = np.zeros(self.inpN)
        # Create list to store weighted sum inputs in feedforward step
        inp = [None] * self.totL
        # Create list of arrays for deltas of this backward pass
        dW = [None] * self.totL
        dB = [None] * self.totL
        # Iteratively populate lists with arrays for each layer
        for l in range(self.totL):
            i = l + 1
            dWThr[l] = np.zeros((self.nod[i],threadSize))
            dBThr[l] = np.zeros((self.nod[i],threadSize))
            val[i] = np.zeros(self.nod[i])
            inp[l] = np.zeros(self.nod[i])
            dW[l] = np.zeros(self.nod[i])
            dB[l] = np.zeros(self.nod[i])
        # Create array for expected outputs
        exp = np.zeros(self.outN, dtype=np.int)

        # Iterate through the data assigned to this thread, training model
        for i in range(threadSize):
            val, exp = self.importDat(im[i], lab[i], val, exp)
            inp, val = self.feedForward(inp, val)
            dW, dB, eThr[i], inp, val, exp = self.backpropogate(inp, val, exp,
                    dW, dB)
            # Populate NN deltas list for update step with each result of batch
            for l in range(self.totL):
                dWThr[l][:,i], dBThr[l][:,i] = dW[l], dB[l]

        
        e = [eThr, dWThr, dBThr]
        return e

    # Validation function to test performance during training
    def validate(self, im, lab):
        runs = len(lab)
        vals = [None] * runs
        costs = np.zeros((runs, self.outN))
        score = 0
        val = [None] * self.totV
        val[0] = np.zeros(self.inpN)
        exp = np.zeros(self.outN, dtype=np.int)
        # Create list to store weighted sum inputs in feedforward step
        inp = [None] * self.totL
        for l in range(self.totL):
            i = l + 1
            val[i] = np.zeros(self.nod[i])
            inp[l] = np.zeros(self.nod[i])
        for i in range(runs):
            val, exp = self.importDat(im[i], lab[i], val, exp)
            inp, val = self.feedForward(inp, val)
            costs[i] = self.logLoss(val[self.totL], exp)
            yHat = np.argmax(val[self.totL])
            y = np.argmax(exp)
            vals[i] = [yHat, y]
            if yHat == y: score += 1

        e = [np.sqrt(np.mean(np.square(costs))), score, vals]
        return e


