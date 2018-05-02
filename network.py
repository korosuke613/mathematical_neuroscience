from typing import List
from layer import Layer
import numpy as np


class Network:
    def __init__(self, input_num_, hidden_layer_num_=None):
        self.hidden_layer_num = hidden_layer_num_
        self.layers: List[Layer] = list()
        self.input_num = input_num_
        self.in_layer: Layer = None
        self.out_layer: Layer = None

    @staticmethod
    def gen_layer(input_num_, perceptron_num,
                  weight_=None, bias_=None):
        layer = Layer(input_num_, perceptron_num)
        layer.set_weight(weight_=weight_)
        layer.set_bias(bias_=bias_)
        return layer

    def gen_hidden_layers(self, input_num_, perceptron_num_):
        next_num = input_num_
        for i in range(self.hidden_layer_num):
            self.layers.append(self.gen_layer(input_num_=next_num, perceptron_num=perceptron_num_))
            next_num = perceptron_num_

    def forward(self, input_):
        self.in_layer.set_input(input_)
        next_input = self.in_layer.get_output_with_sigmoid()

        for hidden_layer in self.layers:
            hidden_layer.set_input(next_input)
            next_input = hidden_layer.get_output_with_sigmoid()

        self.out_layer.set_input(next_input)
        self.out_layer.get_output_with_sigmoid()

        return self.out_layer.output_matrix

    def backward(self, Y, D):
        delta_out = Y - D
        grad_out = self.layers[len(self.layers)-1].output_matrix.T.dot(delta_out)
        self.out_layer.weight_matrix -= 0.05 * grad_out

        up_matrix = self.out_layer.output_matrix
        up_delta = delta_out
        for key, layer in enumerate(reversed(self.layers)):
            key = len(self.layers) - key -1
            if key - 1 >= 0:
                down_matrix = self.layers[key-1].output_matrix
            else:
                down_matrix = self.in_layer.output_matrix
            sigmoid_dash = layer.output_matrix * (1 - layer.output_matrix)

            delta_hidden = up_delta.dot(up_matrix.T) * sigmoid_dash
            grad = down_matrix.T.dot(delta_hidden)
            layer.weight_matrix -= 0.05 * grad
            up_matrix = layer.output_matrix
            up_delta = delta_hidden

        sigmoid_dash = self.in_layer.output_matrix * (1 - self.in_layer.output_matrix)
        delta_in = delta_hidden.dot(self.layers[0].weight_matrix.T) * sigmoid_dash
        X = np.array([self.in_layer.input_matrix])
        grad_in = X.T.dot(delta_in)
        self.in_layer.weight_matrix -= 0.05 * grad_in


if __name__ == '__main__':
    X = np.array([1.0, 0.0])

    W1 = np.array([
        [0.1, 0.3, 0.5],
        [0.2, 0.4, 0.6]
    ])
    B1 = np.array([0.1, 0.2, 0.3])
    W2 = np.array([
        [0.1, 0.4],
        [0.2, 0.5],
        [0.3, 0.6]
    ])
    B2 = np.array([0.1, 0.2])
    W3 = np.array([
        [0.1, 0.3],
        [0.2, 0.4]
    ])
    B3 = np.array([0.1, 0.2])
    network = Network(input_num_=2)
    network.in_layer = network.gen_layer(input_num_=2,
                                         perceptron_num=3,
                                         weight_=W1,
                                         bias_=B1)
    network.layers.append(network.gen_layer(input_num_=3,
                                            perceptron_num=2,
                                            weight_=W2,
                                            bias_=B2))
    network.out_layer = network.gen_layer(input_num_=2,
                                          perceptron_num=2,
                                          weight_=W3,
                                          bias_=B3)
    Y = network.forward(X)

    print(Y)
    print()

    network = Network(input_num_=2, hidden_layer_num_=3)
    network.in_layer = network.gen_layer(input_num_=2, perceptron_num=3)
    network.gen_hidden_layers(input_num_=3, perceptron_num_=2)
    network.out_layer = network.gen_layer(input_num_=2, perceptron_num=2)

    for i in range(1000):
        Y = network.forward(np.array([1.0, 0.0]))
        network.backward(Y, np.array([0.0, 1.0]))
        print(Y)
        Y = network.forward(np.array([0.0, 1.0]))
        network.backward(Y, np.array([1.0, 0.0]))
        print(Y)
        Y = network.forward(np.array([0.0, 0.0]))
        network.backward(Y, np.array([1.0, 1.0]))
        print(Y)
        Y = network.forward(np.array([1.0, 1.0]))
        network.backward(Y, np.array([0.0, 0.0]))
        print(Y)
        print()
