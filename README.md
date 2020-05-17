# Deep Neural Network
Simple deep neural network implementation, built in Python 3 for CPU and GPU processing using Numpy and Cupy respectively.

This implementation uses the [MNIST](http://yann.lecun.com/exdb/mnist/) database to classify handwritten digits, but is flexible and so can be used for other classification tasks with minor architecture changes.

The networks can have an arbitrary architecture, using different activations for each layer.

## Getting started
Experiments are run from a separate script, either `runDnn.py` or `runDnnGPU.py` for the CPU and GPU implementations respectively.

### Configuration
The `hiddenLayers` variable takes a list of nodes which will be the hidden layers of the network, with the input and output layers at either end, specified by the `inputNodes` and `outputNodes` respectively.

The activations of the hidden and output layers can be set after initialisation of the network class by creating a list of the activation functions as `hiddenActivations` and the output activation as `outputActivation`

### Get MNIST data
- Reads files from 'data' directory
  * Get data and decompress manually from:
      http://yann.lecun.com/exdb/mnist/
  * Run script which comes with `python-mnist`:
      - Depending on root or user pip install path:
          * `/bin/mnist_get_data.sh`
          * `~/.local/bin/mnist_get_data.sh`

### Loading and saving models
You will need to create the directory `modelDNN` for the models weights and biases to be saved into, I recommend using a separate directory to store the saved models and load from them by setting the `modelDir` variable.

There are no checks so you will need to make sure the model being loaded matches the architecture set for the model when initialised (this doesn't apply to activation functions).

### Requirements
- Numpy
- Cupy
  * CUDA Toolkit
- mnist (python-mnist)
- Matplotlib

## Numpy CPU implementation
This is designed for use with the `multiprocessing` python module and does so through worker pools. By using the worker pools to break up the training batch size into sub batches the deltas calculated during back-propagation can be collected and averaged once the batch is completed to update the weights and biases of the network. This style of parallelisation requires pickling of the data to return it from each worker, meaning that this implementation gets less efficient the more worker pools are used. This could be optimised by using shared memory to allow for more efficient communication between workers and the main process, as there are no conflicts that could result from race conditions.

The debug experiment script (`runDnnDebug.py`) is a CPU implementation without the use of multiprocessing, to allow for `print` and `input` statements to be used for bug-fixing.

## Cupy GPU implementation
[Cupy](https://cupy.chainer.org/) is an implementation of Numpy utilising the CUDA toolkit. This allows for a much higher throughput of calculations when using larger networks, although for efficient execution all data must be initially loaded into the GPU meaning that this implementation is very dependent in the available VRAM. Cupy doesn't play nicely with multiprocessing and so a multi-threaded implementation of this isn't likely to be possible (let me know of you manage to do it). Although with the usage profile of _my_ GPU it doesn't appear it would help but I could see it being so for much larger networks.
