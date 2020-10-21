import numpy as np

# sigmoid function to normalize inputs


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# sigmoid derivatives to adjust synaptic weights
def sigmoid_derivative(x):
    return x * (1 - x)


# input dataset
training_inputs = np.array([[0, 0, 1, 0, 0, 1, 1],
                            [1, 1, 1, 1, 1, 0, 1],
                            [1, 0, 1, 1, 0, 0, 1],
                            [0, 1, 1, 0, 0, 1, 0],
                            [1, 0, 1, 1, 0, 0, 1],
                            [0, 0, 1, 0, 0, 1, 1]])

# output dataset
training_outputs = np.array([[1, 1, 1, 1, 1, 1]]).T

# seed random numbers to make calculation
np.random.seed(1)

# initialize weights randomly with mean 0 to create weight matrix, synaptic weights
synaptic_weights_0 = 2 * np.random.random((7, 2)) - 1
synaptic_weights_1 = 2 * np.random.random((2, 1)) - 1

print('Random starting synaptic weights first: ')

print(synaptic_weights_0)

print('Random starting synaptic weights second: ')

print(synaptic_weights_1)

# Iterate 10,000 times
for iteration in range(100000):
    # Define input layer
    input_layer = training_inputs
    # Normalize the product of the input layer with the synaptic weights
    outputs_0 = sigmoid(np.dot(input_layer, synaptic_weights_0))
    outputs_1 = sigmoid(np.dot(outputs_0, synaptic_weights_1))

    # how much did we miss?
    error = training_outputs - outputs_1

    # multiply how much we missed by the
    # slope of the sigmoid at the values in outputs
    adjustments = error * sigmoid_derivative(outputs_1)

    # update weights
    synaptic_weights_0 += np.dot(input_layer.T, adjustments)
    synaptic_weights_1 += np.dot(outputs_0.T, adjustments)

print('Synaptic weights after training: ')
print('Primeira Camada:')
print(synaptic_weights_0)
print('Segunda Camada:')
print(synaptic_weights_1)
print("Output After Training:")
print('Primeira Camada:')
print(outputs_0)
print('Segunda Camada:')
print(outputs_1)
