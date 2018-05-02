import numpy
import matplotlib.pyplot as plt
from plotdecisionresions import plot_decision_regions
from neural_network import NeuralNetwork


def generate_mirror(digit):
    def generate_binary_with_mirror(decimal, digit_):
        binary = generate_binary(decimal, digit_)
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


def convert_mirror_to_ndarray(mirror_):
    def convert_string_ndarray(string):
        string = [int(s) for s in list(string)]
        return string

    array = numpy.array([])
    y = numpy.array([])
    for m in mirror_:
        array = numpy.append(array, convert_string_ndarray(m["binary"]))
        y = numpy.append(y, m["is_mirror"])
    digit = len(mirror_[0]["binary"])
    array = numpy.reshape(array, (2**digit, digit))

    return array, y


def main():
    numpy.random.seed(0)

    # Initialize the NeuralNetwork with
    # 2 input neurons
    # 2 hidden neurons
    # 1 output neuron

    # Set the input data
    digit = 6
    nn = NeuralNetwork([digit, 3, 2, 3, 1])
    mirror = generate_mirror(digit)
    X, y = convert_mirror_to_ndarray(mirror)

    # Set the labels, the correct results for the xor operation

    epoch_num = 1000
    nn.steps_per_epoch = 2 ** digit
    epochs = [x for x in range(epoch_num)]
    errors = nn.fit(X, y, epochs=epoch_num)

    # Show the prediction results
    print("Final prediction")
    for s in X:
        print(s, nn.predict_single_data(s))

#    plot_decision_regions(numpy.array([X]), y, nn)
#    plt.xlabel('x-axis')
#    plt.ylabel('y-axis')
#    plt.legend(loc='upper left')
#    plt.tight_layout()
#    plt.show()

    plt.xlabel('epochs')
    plt.ylabel('error')
    plt.plot(epochs, errors)
    plt.plot(epochs, numpy.poly1d(numpy.polyfit(epochs, errors, 4))(epochs))
    plt.show()


if __name__ == '__main__':
    main()
