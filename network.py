from typing import List
from layer import Layer
import numpy as np


class Network:
    def __init__(self, layer_num_, input_num_):
        self.layer_num = layer_num_
        self.layers: List[Layer] = list()
        self.input_num = input_num_
        self.in_layer: Layer = None
        self.out_layer: Layer = None

    def gen_layer(self, perceptron_num,
                  weight_=None, bias_=None):
        layer = Layer(self.input_num, perceptron_num)
        layer.set_weight(weight_=weight_)
        layer.set_bias(bias_=bias_)
        return layer

    def forward(self, input_):
        self.in_layer.set_input(input_)
        next_input = self.in_layer.get_output_with_sigmoid()

        for hidden_layer in self.layers:
            hidden_layer.set_input(next_input)
            next_input = hidden_layer.get_output_with_sigmoid()

        self.out_layer.set_input(next_input)
        self.out_layer.get_output_with_identity()

        return self.out_layer.output_matrix


if __name__ == '__main__':
    X = np.array([1.0, 0.5])

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
    network = Network(3, 2)
    network.in_layer = network.gen_layer(perceptron_num=3,
                                         weight_=W1,
                                         bias_=B1)
    network.layers.append(network.gen_layer(perceptron_num=2,
                                            weight_=W2,
                                            bias_=B2))
    network.out_layer = network.gen_layer(perceptron_num=2,
                                          weight_=W3,
                                          bias_=B3)
    Y = network.forward(X)

    print(Y)