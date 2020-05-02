# Neural Network from scratch

## This repository is made for the sole purpose of implementing and showing how to implement Neural Network from scratch
### For this I have used the dataset of a bank which contains the data about different customers and whether they left the bank or not

### For implementing a Neural Network, there are 3 main steps:
#### Forward Propagation
#### Computing cost
#### Backward Propagation

# Image of a neural network

![](images/neural_network.png)

##### Here the input layer contains 3 units, 1st hidden layer contains 4 units, 2nd hidden layer contains 4 units and the output layer contains 1 unit. This is a 3 layer network because we don't count input layer as a layer.

#### Forward Propagation
Forward Propagation is nothing but series of matrix multiplications along the depth of the network, where the weights of current layer is taken dot product with the activation matrix of previous layers.

##### Weight matrix
What is a weight matrix? A weight matrix is nothing but a matrix of random numbers with shape of (current, previous) where current is the number of units/nodes(A unit is a single neuron in a network) in the current layers(for which you are calculating the activations and previous is the number of nodes/units from the previous activation layer)
For example, if your network have a 3 layer architecture with [3, 4, 4, 1] nodes. [3, 4, 4, 1] means 3 nodes in input layer, 4 nodes in the first hidden layer, 4 nodes in the second hidden layer and 1 node in the output layer.
Now, as you have noticed there are 4 elements in the array. That's because we don't count input layer as a seperate layer. So, if we want to find number of layers in a neural network then it would be equal to number of hidden layers and output layer.

So, if we go by this example W1 will have shape (4, 3), W2 will have shape (4, 4) and W3 will have shape (1, 4) and each element of these matrices will be initialized with small random values. We never take W0 as there is no point of taking a weight which is giving output to input layer.

For finding the activation/output of first hidden layer, we have to do the dot product between the W1(first layer weight matrix) and (X)input layer(which is the previous layer). So the output Z1[1] will be W1.X. If bias(b) is also added then it would be equal to W1.X + b.
Zn[m] -->  output of nth node in layer m
Z1[1] -->  output of 1st node in layer 1
Z2[1] -->  output of 2nd node in layer 1
Z3[1] -->  output of 3rd node in layer 1

![](images/Z1.PNG)

So,
              Z1 = W1.X + b
Similarly,
              Z2 = W2.A1 + b
              Z3 = W3.A2 + b
We shall see A1, A2, A3 later when we see about activation functions.

#### Activation Functions:
Every output of a node is then passed through an activation function. Most of the times, non linear activation functions are used. Activation functions are a very important part of neural network. It helps in computing the result in desirable form and also helps in improving accuracy of our model.
There are many types of activation functions but some of the popular activation functions are:
1. Threshold Activation Function
2. Sigmoid Function
3. Tanh Activation Function
4. Rectified Linear Unit Function(ReLU)
