import numpy as np
from abc import ABC


def sigmoid(u):
    return 1 / (1 + np.exp(-u))


class Perceptron:
    def __init__(self, param_):
        self.param = param_


class Layer(ABC):
    def __init__(self):
        self.neuron = 0.0
        self.b = 0.0
        self.weight = 0.0


class Network:
    def __init__(self, input_array_):
        self.i_layer = InputLayer(input_array_)
        self.h_layer = HiddenLayer()
        self.o_layer = OutputLayer()
        self.learning_rate = 0.05
        self.delta = list()

    def check(self, binary):
        self.i_layer.neuron = np.array([float(binary[0]),
                                        float(binary[1]),
                                        float(binary[2]),
                                        float(binary[3]),
                                        float(binary[4]),
                                        float(binary[5])
                                        ])
        y, _ = self.forward()
        print(f"ミラーの確立は{y}です")

    def get_delta(self):
        total = sum(self.delta)
        self.delta = list()
        return total

    def forward(self):
        W1 = self.i_layer.weight
        W2 = self.h_layer.weight
        X = self.i_layer.neuron

        U1 = X.dot(W1) + self.i_layer.b
        Z1 = sigmoid(U1)
        U2 = Z1.T.dot(W2.T) + self.h_layer.b
        y = sigmoid(U2)

        self.h_layer.neuron = Z1
        self.o_layer.neuron = y

        return y, Z1

    def back_propagation(self):
        W1 = self.i_layer.weight
        W2 = self.h_layer.weight
        y = self.o_layer.neuron
        Z1 = self.h_layer.neuron
        X = self.i_layer.neuron

        if self.i_layer.correct is True:
            d = 1.0
        else:
            d = 0.0

        delta2 = y - d

        self.delta.append(delta2**2)

        grad_W2 = Z1.dot(delta2)

        sigmoid_dash = Z1 * (1 - Z1)

        delta1 = W2.T.dot(delta2.T) * sigmoid_dash
        X = np.array([X])
        grad_W1 = X.T.dot(delta1)

        W2 -= self.learning_rate * grad_W2.T
        W1 -= self.learning_rate * grad_W1


class InputLayer(Layer):
    def __init__(self, input_array: dict):
        super().__init__()
        self.correct = input_array["is_mirror"]
        self.x1 = Perceptron(float(input_array["binary"][0]))
        self.x2 = Perceptron(float(input_array["binary"][1]))
        self.x3 = Perceptron(float(input_array["binary"][2]))
        self.x4 = Perceptron(float(input_array["binary"][3]))
        self.x5 = Perceptron(float(input_array["binary"][4]))
        self.x6 = Perceptron(float(input_array["binary"][5]))
        self.b = 0.01 * np.random.randn(1, 2)
        self.weight = 0.01 * np.random.randn(6, 2)
        self.neuron = np.array([self.x1.param,
                                self.x2.param,
                                self.x3.param,
                                self.x4.param,
                                self.x5.param,
                                self.x6.param
                                ])

    def set_array(self, input_array: dict):
        self.correct = input_array["is_mirror"]
        self.x1 = Perceptron(float(input_array["binary"][0]))
        self.x2 = Perceptron(float(input_array["binary"][1]))
        self.x3 = Perceptron(float(input_array["binary"][2]))
        self.x4 = Perceptron(float(input_array["binary"][3]))
        self.x5 = Perceptron(float(input_array["binary"][4]))
        self.x6 = Perceptron(float(input_array["binary"][5]))
        self.neuron = np.array([self.x1.param,
                                self.x2.param,
                                self.x3.param,
                                self.x4.param,
                                self.x5.param,
                                self.x6.param
                                ])


class HiddenLayer(Layer):
    def __init__(self):
        super().__init__()
        self.u1 = Perceptron(0.0)
        self.u2 = Perceptron(0.0)
        self.b = np.array([[0.2, 0.2]])
        self.weight = 0.01 * np.random.randn(2, 1)
        #self.w = np.array([[0.1, 0.2]])


class OutputLayer(Layer):
    def __init__(self):
        super().__init__()
        self.y = 0.0


def main():
    # 0 例題データの生成
    mirror = generate_mirror(6)

    # 0 結合係数の初期設定
    #init_param()

    new = Network(mirror[0])
    for epoch in range(0, 1000):
        # 1. 回路に入力を与える
        new.i_layer.set_array(mirror[epoch % 64])
        new.forward()
        new.back_propagation()
        print(f"epoch:{epoch}, loss:{new.get_delta()}")


def init_param():
    pass


def generate_mirror(digit):
    def generate_binary_with_mirror(decimal, digit):
        binary = generate_binary(decimal, digit)
        is_mirror = False
        half_digit = digit // 2
        if binary[:half_digit] == binary[-half_digit:][::-1]:
            is_mirror = True
        return {"binary": binary, "is_mirror": is_mirror}

    def generate_binary(decimal_, digit_):
        binary = format(decimal_, 'b')
        binary = f"{binary:0>14}"[-digit_:]
        return binary

    # if digit % 2 != 0:
    #    raise ValueError("The number of digits must be an even number.")

    binaries = [generate_binary_with_mirror(decimal, digit)
                for decimal in range(2**digit)]
    return binaries


if __name__ == "__main__":
    #print(generate_mirror(6))
    #input_layer = InputLayer(generate_mirror(6)[13])
    #print(input_layer)
    main()
