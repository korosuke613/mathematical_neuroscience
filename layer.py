import numpy as np


def identity_function(x):
    return x


def sigmoid(x):
    """シグモイド関数"""
    return 1 / (1 + np.exp(-x))


class Layer:
    def __init__(self, input_num_, perceptron_num_):
        self.input_num = input_num_
        self.perceptron_num = perceptron_num_
        self.input_matrix = None
        self.weight_matrix = None
        self.bias = None
        self.output_matrix = None
        self.ADD_BIAS = 1

    def set_input(self, input_):
        self.input_matrix = input_

    def set_weight(self, weight_=None):
        if weight_ is None:
            self.weight_matrix = 0.01 * np.random.randn(self.input_num, self.perceptron_num)
            return
        self.weight_matrix = weight_

    def set_bias(self, bias_=None):
        if bias_ is None:
            self.bias = 0.01 * np.random.randn(1, self.perceptron_num)
            return
        self.bias = bias_

    def get_output(self):
        output = np.dot(self.input_matrix, self.weight_matrix) + self.bias
        return output

    def get_output_with_sigmoid(self):
        output = sigmoid(self.get_output())
        self.output_matrix = output
        return output

    def get_output_with_identity(self):
        output = identity_function(self.get_output())
        self.output_matrix = output
        return output


def main():
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

    in_layer = Layer(input_num_=2, perceptron_num_=3)
    in_layer.set_input(X)
    in_layer.set_weight(W1)
    in_layer.set_bias(B1)
    in_layer.get_output_with_sigmoid()

    hidden_layer = Layer(input_num_=3, perceptron_num_=2)
    hidden_layer.set_input(in_layer.output_matrix)
    hidden_layer.set_weight(W2)
    hidden_layer.set_bias(B2)
    hidden_layer.get_output_with_sigmoid()

    out_layer = Layer(input_num_=2, perceptron_num_=2)
    out_layer.set_input(hidden_layer.output_matrix)
    out_layer.set_weight(W3)
    out_layer.set_bias(B3)
    out_layer.get_output_with_identity()

    print(out_layer.output_matrix)


if __name__ == '__main__':
    main()
