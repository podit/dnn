'''
    Deep feed-forward neural network implemented using Cupy - Python 3.8
    
        - Performs forward and backward propogation through an abritrary
            layer architecture formed from the supplied nodes for each layer
        - Allows each layer to use an individual activation function:
            - Sigmoid, Tanh, ReLU, ELU, Softplus, Softmax
        - Built for use with the Python cupy module
        - Specifically designed for large networks using GPU computation
'''

import numpy as np
import numpy.random as npr

import cupy as cp
import cupy.random as cpr

class DNN (object):
    # Initialise neural network parameters
    def __init__(self, hidNod, inpNod, outNod, lrnRat, batchSize):
        # Set learning rate
        self.lrnR = lrnRat
        # Set batch size
        self.btc = batchSize
        self.btcCount = 0
        
        # TODO: Consider integration into class initialisation inputs
        # Set alpha parameter for use in ELU activation function
        self.alpha = .1
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
        
        # Create list for node values and initialise the input layer array
        self.val = [None] * self.totV
        self.val[0] = cp.zeros(self.inpN)
        # Create list to store weighted sum inputs in feedforward step
        self.inp = [None] * self.totL

        # TODO: Integrate weights and bias initialisation ranges into init
        # Iterate through connected layers creating training arrays
        for l in range(self.totL):
            i = l + 1
            self.wei[l] = cpr.uniform(-1,1,(self.nod[i],self.nod[l]))
            self.bis[l] = cpr.uniform(-1,1,self.nod[i])
            self.dummyBis[l] = cp.zeros(self.nod[i])
            # Create delta arrays corresponding to the batch size for update
            self.dW[l] = cp.zeros((self.nod[i], self.btc))
            self.dB[l] = cp.zeros((self.nod[i], self.btc))
            # Create arrays for each node's value and input
            self.val[i] = cp.zeros(self.nod[i])
            self.inp[l] = cp.zeros(self.nod[i])
        # Create array for expected outputs
        self.exp = cp.zeros(self.outN, dtype=cp.int)

        # Display the initialised network
        self.dim()
    
    def reinitialise(self):
        for l in range(self.totL):
            i = l + 1
            self.wei[l] = cpr.uniform(-1,1,(self.nod[i],self.nod[l]))
            self.bis[l] = cpr.uniform(-1,1,self.nod[i])

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
        for i, l in enumerate(self.wei): dims[i] = l.shape
        return dims

    # Function to save model
    def modelS(self):
        # Create template file names
        wfname = 'modelDNN/wei{:d}'
        bfname = 'modelDNN/bis{:d}'
        # Iterate through weights list and save arrays
        for w in range(len(self.wei)): cp.save(wfname.format(w), self.wei[w])
        # Iterate through biases list and save arrays
        if self.biases:
            for b in range(len(self.bis)):
                cp.save(bfname.format(b), self.bis[b])
   
    # Function to load model from specified directory
    def modelL(self, modelDir):
        # Create template file names
        wfname = modelDir + '/wei{:d}.npy'
        bfname = modelDir + '/bis{:d}.npy'
        # Iterate through weights list and load arrays
        for w in range(len(self.wei)): self.wei[w] = cp.load(wfname.format(w))
        # Iterate through biases list and load arrays
        if self.biases:
            for b in range(len(self.bis)):
                self.bis[w] = cp.load(bfname.format(b))
    
    # Function to import input data into the NN
    def importDat(self, img, lab):
        self.val[0] = img
        self.exp = lab

    # Activation functions (pass prime as True for derivative)
    def sigmoid(self, x, prime=False): 
        if prime: 
            sig = self.sigmoid(x)
            return sig * (1 - sig)
        else: return 1 / (1 + cp.exp(-x))

    def tanh(self, x, prime=False):
        if prime: return cp.subtract(1, self.tanh(x)**2)
        else: return cp.multiply(1.7159, cp.tanh((2/3) * x))

    # TODO: np.where functions are backwards, see crossEntropy func
    def relu(self, x, prime=False):
        if prime:
            x = cp.where(x < 0, x, 1)
            x = cp.where(x == 1, x, 0)
            return x
        else:
            x = cp.where(x >= 0, x, 0)
            return x
    def elu(self, x, prime=False):
        if prime:
            x = cp.where(x < 0, x, 1)
            x = cp.where(x == 1, x, self.alpha * cp.exp(x))
            return x
        else:
            x = cp.where(x >= 0, x, self.alpha * (cp.exp(x) - 1))
            return x

    def softplus(self, x, prime=False):
        if prime: return 1 / (1 + np.exp(-x))
        else: return np.log(1 + np.exp(x))

    def softmax(self, x, prime=False):
        if prime:
            xm = x.reshape((-1,1))
            return cp.diagflat(x) - cp.dot(xm, xm.T)
        else:
            x -= cp.max(x)
            return (cp.exp(x) / cp.sum(cp.exp(x), axis=0)).T
    
    # TODO: Reconsider implementations of more advanced loss functions
    # Calculate hinge loss
    def hingeLoss(self, yHat, y):
        return cp.maximum(0, 1 - yHat * y)
    # Calculate cross entropy
    def crossEntropy(self, yHat, y):
        err = cp.zeros(y.size)
        err = cp.where(y == 1, -cp.log(yHat), yHat)
        err = cp.where(y == 0, -cp.log(1 - yHat), err)
        return err

    # Loss functions used for backpropogation error calculation
    def loss(self, yHat, y):
        return cp.subtract(yHat, y)
    def logLoss(self, yHat, y):
        return cp.multiply(y, cp.log(yHat))

    # Process the input through the NN
    def feedForward(self):
        # Define output layer for cleaner indexing
        o = self.totL - 1

        # Iterate through hidden Layers
        for h in range(self.hidLrs):
            # Calculate weighted inputs for layer connections
            weightedInputs = cp.multiply(self.val[h], self.wei[h])
            # Get sum of weighted inputs for each node and add node bias
            weightedSum = cp.add(cp.sum(weightedInputs, axis=1),
                    self.bis[h] if self.biases else self.dummyBis[h])
            # Record weighted sum for backpropogation step
            self.inp[h] = weightedSum
            
            # Set node value using the activation function and weighted sum
            self.val[h+1] = self.hidActs[h](weightedSum)

        # Output Layer calculation (same process as hidden layers)
        weightedInputs = cp.multiply(self.val[o], self.wei[o])
        weightedSum = cp.add(cp.sum(weightedInputs, axis=1),
                self.bis[o] if self.biases else self.dummyBis[o])
        self.inp[o] = weightedSum
        
        self.val[self.totL] = self.outAct(weightedSum)

    def backpropogate(self):
        # Define output layer index
        o = self.totL - 1
        s = self.btcCount
        self.btcCount += 1
        
        # Get loss for use in calculation of deltas
        loss = self.logLoss(self.val[self.totL], self.exp)
        
        # Calculate output delta using the cost (error) and derivative of output
        #   layer activation function applied to the weighted sum for the layer
        # - If output layer activation is softmax the cost must be multiplied by
        #   the derivative matrix and summed to the error gradient vector
        if self.outAct == self.softmax: delta = cp.sum(cp.multiply(loss,
            self.outAct(self.val[self.totL], True)), axis=1)
        else: delta = cp.multiply(loss, self.outAct(self.inp[o], True))
        # Record deltas for weights and biases of the output layer
        self.dW[o][:,s] = delta
        if self.biases: self.dB[o][:,s] = delta
       
       # Backpropogate the delta for the output layers through the weighted
       #    connections of each hidden layer
        for h in reversed(range(self.hidLrs)):
            # Calculate layer delta by multiplying the delta of the previous
            #   layer through the connected weights
            delta = cp.multiply(cp.dot(delta, self.wei[h+1]),
                    self.hidActs[h](self.inp[h], True))
            # Record deltas for weights and biases through each layer
            self.dW[h][:,s] = delta
            if self.biases: self.dB[h][:,s] = delta
        
        # Calculate error across all output nodes for profiling training
        error = np.sqrt(np.mean(np.square(cp.asnumpy(loss))))

        # Return deltas for use in batch update (return arrays for re-use)
        return error

    # Function to update weights and biases upon batch completeion
    def update(self):
        o = self.totL - 1
        self.btcCount = 0
        # Update weights and biases for hidden layers using deltas
        for h in range(self.hidLrs):
            self.wei[h] = cp.transpose(cp.subtract(self.wei[h].T,
                    cp.multiply(self.lrnR, cp.mean(self.dW[h], 1))))
            if self.biases: self.bis[h] -= cp.multiply(self.lrnR,
                    cp.mean(self.dB[h], 1))
        # Update weights and biases for output layer using deltas
        self.wei[o] = cp.transpose(cp.subtract(self.wei[o].T,
                cp.multiply(self.lrnR, cp.mean(self.dW[o], 1))))
        if self.biases: self.bis[o] -= cp.multiply(self.lrnR,
                cp.mean(self.dB[o], 1))
    
    # Function to control training (optimised for single thread with GPU)
    def run(self, im, lab):
        # Create numpy array to store errors
        e = np.zeros(self.btc)

        # Iterate through the data assigned to this thread, training model
        for i in range(self.btc):
            self.importDat(im[i], lab[i])
            self.feedForward()
            e[i] = self.backpropogate()

        return e

    # Validation function to test performance during training
    def validate(self, im, lab):
        runs = len(lab)
        vals = [None] * runs
        costs = cp.zeros((runs, self.outN))
        score = 0
        # Perform each feed forward step and record loss and classifications
        for i in range(runs):
            self.importDat(im[i], lab[i])
            self.feedForward()
            costs[i] = self.logLoss(self.val[self.totL], self.exp)
            yHat = cp.asnumpy(self.val[self.totL].argmax())
            y = cp.asnumpy(self.exp.argmax())
            vals[i] = [yHat, y]
            if yHat == y: score += 1

        return np.sqrt(np.mean(np.square(cp.asnumpy(costs)))), score, vals


