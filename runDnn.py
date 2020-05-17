'''
    Control script for experiment utilising the CPU optimised DNN class
    TODO: Plot classification accuracies of each digit through training
'''

from dnn import *

import multiprocessing as mp
from timeit import default_timer as timer
import matplotlib.pyplot as plt
# Running mean function used for plotting
def runningMean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

from importMNIST import imp
dat = imp()

# Set flags to load and save model (model must be same size as loaded model)
loadModel = False
modelDir = 'models/DNN/800-42.20'
saveModel = False

# Model structure parameters
hiddenLayers = [400,200]
inputNodes = 784
outputNodes = 10

# Import data normalised between 0 and 1 or -1 and 1 (vanishing mean)
vanishingMean = False

# Disable node biases for testing
testBiases = False

# Set number of training epochs
epochs = 64
check = int(epochs/8)

# TODO: Change name of 'momentum' as momentum is already a concept in ML
# Set length of 'momentum metric' (compares previous mean accuracies to latest
momLen = 5      #   mean accuracies, proportional to the length)
N = momLen      # Set length of rolling averages, for plots, to momentum length

# Set learning rate magnitude and calculate learning rate
magLrnRate = -3
stochasticLearningRate = 10 ** magLrnRate
targetMagLrnRate = -5
targetStochasticLearningRate = 10 ** targetMagLrnRate

# Set absolute size of training data set and the batch multiplier
absoluteSize = 60000
batchMult = 20
# Set number of workers (threads) to be used
threadNum = 5

# Calculate batch and thread size
batchSize = batchMult * threadNum
threadSize = int(batchSize/threadNum)

# Calculate learning rate based on batch size
learningRate = batchSize * stochasticLearningRate
targetLearningRate = batchSize * targetStochasticLearningRate
learningRateDelta = (targetLearningRate - learningRate) / epochs

# Calculate number of runs per epoch based on batch size and worker pool
runs = int((absoluteSize / threadSize) / threadNum)

print(hiddenLayers)
input('Waiting for execution consent...')

# Initialise the neural network with node numbers, learning rate
#   and batch size
net = DNN(hiddenLayers, inputNodes, outputNodes, learningRate, batchSize)
# Activation functions for hidden and output layers of model
hiddenActivations = [net.sigmoid] * len(hiddenLayers)
outputActivation = net.softmax
# Set the activation functions to be used in the model class
net.setActivation(hiddenActivations, outputActivation)

# Create lists to unpack deltas and errors from pool workers
dW = [None] * threadNum
dB = [None] * threadNum
e = [None] * threadNum
# Create array to store errors for each run
errors = np.zeros(batchSize)

# Create array for recording mean error per run
runErr = np.zeros((epochs, runs))

# Create trace array for recording performance through training process
trace = np.zeros(epochs)
traceVal = np.zeros(epochs)
traceAcc = np.zeros(epochs)

# Import training data and get length of dataset
imTr, labTr = dat.importTrain(vanishingMean)
rng = len(labTr)
# Import validation set
imTs, labTs = dat.importTest(vanishingMean)
rngTs = len(labTs)

# Load model weights and biases from file
if loadModel: net.modelL(modelDir)

execStart = timer()
workableInit = False
while not workableInit:
    # Iterate through each epoch
    for i in range(epochs):
        epStart = timer()
        # Create pool of workers
        pool = mp.Pool(threadNum)
        
        # Created randomised array of indexes for the input data
        ind = npr.permutation(rng)
        indVl = npr.permutation(rngTs)
        # Slice randomised indexes into the minibatches for each thread and run
        thrInd = np.split(ind, runs * threadNum)
        thrIndVl = np.split(indVl, threadNum)
        
        # Iterate through each run
        for j in range(runs):
            # Create list of variables to be loaded into each worker
            runVars = []
            for k in range(threadNum):
                # Calculate index for sliced indexes
                r = (j*k) + k
                # Create list of variables for worker
                inputs = []
                # Add thread size
                inputs.append(k)
                inputs.append(threadSize)
                # Add input images and output labels (using sliced indexes)
                inputs.append(imTr[thrInd[r]])
                inputs.append(labTr[thrInd[r]])
                # Append woker arguments to list
                runVars.append(inputs)
            # Pass arguments to worker pool and wait for results
            e = pool.starmap(net.run, runVars)
            
            # Iteratively write the unpacked results into corresponding arrays
            for t in range(threadNum):
                # Calculate starting and ending indexes for thread size
                start = t * threadSize
                end = start + threadSize
                # Write errors into array
                errors[start:end] = e[t][0][:]
                # TODO: Move these variables to memory shared by worker pool
                # Write deltas for each run into NN object for update step
                for l in range(net.totL):
                    net.dW[l][:,start:end], net.dB[l][:,start:end] = e[t][1][l], e[t][2][l]
            
            # Perform update for this batch
            net.update()
            # Record mean error for batch
            runErr[i][j] = np.mean(errors)

        # Create list of variables to be loaded into each worker
        runVars = []
        for k in range(threadNum):
            # Create list of variables for worker
            inputs = []
            inputs.append(imTs[thrIndVl[k]])
            inputs.append(labTs[thrIndVl[k]])
            # Append woker arguments to list
            runVars.append(inputs)

        e = pool.starmap(net.validate, runVars)

        costVal, scoreVal, vals = 0, 0, []
        for x in e:
            costVal += x[0]
            scoreVal += x[1]
            vals += x[2]
        corr = [0] * 10
        for n in range(rngTs):
            if vals[n][0] == vals[n][1]: corr[vals[n][0]] += 1

        traceVal[i] = costVal / threadNum
        traceAcc[i] = scoreVal / rngTs
        # Calculate momentum (mean of previous accuracies compared to current)
        if i >= 2 * momLen: diff = np.mean(traceAcc[i-momLen:i])\
                - np.mean(traceAcc[i-2*momLen:i-momLen])
        else: diff = float("NAN")

        # Close worker pool once epoch is complete
        pool.close()
            
        epEnd = timer()
        net.lrnR += learningRateDelta
        # Record trace of performance per epoch and report
        trace[i] = np.mean(runErr[i])
        #0 	trn: 0.34 - val: 0.37 0.10 mom: 2.32 - ep end: 14.24
        print(i, '\ttrn: %.2f'% trace[i], '-',
                'val: %.2f %.2f'% (traceVal[i],traceAcc[i]),
                'mom: %.2f' % diff, '-',
                'ep end: %.2f'% (epEnd-epStart))
        #print([0 if c == 0 else 1 for c in corr])
        if not workableInit and i == check and any([c == 0 for c in corr]):
            net.reinitialise()
            net.lrnR = learningRate
            print('reinitialising...', [c == 0 for c in corr])
            break
        if not workableInit and i > check: workableInit = True


# Set score counter to record number of successful classifications
score = 0
# Create list to record predicted and real classifications
vals = [None] * rngTs

# Create list for node values and initialise the input layer array
val = [None] * net.totV
val[0] = np.zeros(net.inpN)
# Create list to store weighted sum inputs in feedforward step
inp = [None] * net.totL
# Create list of arrays for deltas of this backward pass
dW = [None] * net.totL
dB = [None] * net.totL
# Iteratively populate lists with arrays for each layer
for l in range(net.totL):
    i = l + 1
    val[i] = np.zeros(net.nod[i])
    inp[l] = np.zeros(net.nod[i])
    dW[l] = np.zeros(net.nod[i])
    dB[l] = np.zeros(net.nod[i])
# Create array for expected outputs
exp = np.zeros(net.outN, dtype=np.int)

# Do not use biases for testing if flag is set
if testBiases: net.biases = False

# Run the test data throigh the network
for n in range(rngTs):
    val, exp = net.importDat(imTs[n], labTs[n], val, exp)
    inp, val = net.feedForward(inp, val)
    # Get predicted and real clasifications and record
    yHat = np.argmax(val[net.totL])
    y = np.argmax(exp)
    vals[n] = [yHat, y]
    # Check if classification was corrct and update score
    if yHat == y: score += 1

# Populate list with the counts of correct classifications of each digit
corr = [0] * 10
for n in range(rngTs):
    if vals[n][0] == vals[n][1]: corr[vals[n][0]] += 1

execEnd = timer()
execTime = execEnd - execStart
if execTime <= 60: unit = 's'
elif execTime <= 3600: execTime, unit = execTime/60, 'm'
else: execTime, unit = (execTime/60)/60, 'h'
print('Execution time: %.2f%s'% (execTime, unit))
# Report results from validation
# Minimum and final loss and acciracies for training
print('Trn loss: minimum = %.5f final = %.5f'% (trace.min(), trace[-1]))
print('Val loss: minimum = %.5f final = %.5f'% (traceVal.min(), traceVal[-1]))
print('Accuracy: maximum = %.5f final = %.5f'% (traceAcc.max(), traceAcc[-1]))
# Test accuracies
print('Test acc: ', score, '/', rngTs, '\t', (score/rngTs))
print('Digit classification accuracies (0-9):\n', [c/(rngTs/10) for c in corr])
# Network parameters and architecture
print('Learning rate =', stochasticLearningRate, 'to',
        targetStochasticLearningRate, 'Batch size =', batchSize)
net.dim()

# Plot loss and accuracies through training process
plt.plot(runningMean(trace, N))
plt.plot(runningMean(traceVal, N))
plt.plot(runningMean(traceAcc, N))
y = input('Show training profile? `y`')
if y == 'y': plt.show()
# Save model weights and biases to file
if saveModel: net.modelS()
else:
    y = input('Save model? `y`')
    if y == 'y': net.modelS()
