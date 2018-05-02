import numpy as np
from network import Network
import random


def main():
    network = Network(input_num_=1, hidden_layer_num_=3)
    network.in_layer = network.gen_layer(input_num_=1, perceptron_num=3)
    network.gen_hidden_layers(input_num_=3, perceptron_num_=2)
    network.out_layer = network.gen_layer(input_num_=2, perceptron_num=1)

    for i in range(1000):
        ran = i % 10 #float(random.randrange(100))
        Y = network.forward(ran)
        network.backward(Y, ran)
        print(f"ran:{ran}, Y:{Y}")
        print()

    Y = network.forward(33)
    print(f"ran:3, Y:{Y}")


if __name__ == '__main__':
    main()