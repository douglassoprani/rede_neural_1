import numpy as np


class NeuralNetwork():

    def __init__(self):
        np.random.seed(1)

        self.synaptic_weights = 2 * np.random.random((7, 1)) - 1

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def saida(self, inputs):
        inputs = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, self.synaptic_weights))
        return output

    def treino(self, training_inputs, training_outputs, training_iterations):

        for iteration in range(training_iterations):

            output = self.saida(training_inputs)

            error = training_outputs - output

            adjustments = np.dot(training_inputs.T, error*self.sigmoid_derivative(output))

            self.synaptic_weights = self.synaptic_weights+adjustments


if __name__ == "__main__":

    neural_network = NeuralNetwork()

    print("pesos randomicos antes do treinamento")
    print(neural_network.synaptic_weights)

    training_inputs = np.array([[0,0,1,0,0,1,1],
                                [1,1,1,0,1,0,1],
                                [1,0,1,1,0,1,0],
                                [0,1,1,1,0,1,1],
                                [0,1,1,1,0,1,1],
                                [1,1,1,0,0,1,1]])

    training_outputs = np.array([[0,1,1,0,1,0]]).T

    training_iterations = 50000

    neural_network.treino(training_inputs, training_outputs, training_iterations)

    print("pesos treinados após o treinamento")
    print(neural_network.synaptic_weights)

    A = str(input("entrada 1: "))
    B = str(input("entrada 2: "))
    C = str(input("entrada 3: "))
    D = str(input("entrada 4: "))
    E = str(input("entrada 5: "))
    F = str(input("entrada 6: "))
    G = str(input("entrada 7: "))


    print("Nova entrada eh: ", A, B, C, D, E, F, G)
    print("classificacao da rede: ")
    print(neural_network.saida(np.array([A,B,C,D,E,F,G])))







