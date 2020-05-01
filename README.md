# Neural Network from scratch

## This repository is made for the sole purpose of implementing and showing how to implement Neural Network from scratch
### For this I have used the dataset of a bank which contains the data about different customers and whether they left the bank or not

### For implementing a Neural Network, there are 3 main steps:
#### Forward Propagation
#### Computing cost
#### Backward Propagation

#### Forward Propagation
Forward Propagation is nothing but series of matrix multiplications along the depth of the network, where the weights of current layer is taken dot product with the activation matrix of previous layers.

##### Weight matrix
What is a weight matrix? A weight matrix is nothing but a matrix of random numbers with shape of (current, previous) where current is the number of units/nodes(A unit is a single neuron in a network) in the current layers(for which you are calculating the activations and previous is the number of nodes/units from the previous activation layer)
For example, if your network have a 3 layer architecture with [6, 3, 4, 1] nodes. [6, 3, 4, 5] means 6 nodes in input layer, 3 nodes in the first hidden layer, 4 nodes in the second hidden layer and 1 node in the output layer.
Now, as you have noticed there are 4 elements in the array. That's because we don't count input layer as a seperate layer. So, if we want to find number of layers in a neural network then it would be equal to number of hidden layers and output layer.

So, if we go by this example W1 will have shape (3, 6), W2 will have shape (4, 3) and W3 will have shape (1, 4) and each element of these matrices will be initialized with small random values.
