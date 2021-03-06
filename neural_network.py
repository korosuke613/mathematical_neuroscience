import numpy
import matplotlib.pyplot as plt
from plotdecisionresions import plot_decision_regions
from tqdm import tqdm

# The following code is used for hiding the warnings and make this notebook clearer.
import warnings
warnings.filterwarnings('ignore')


def tanh(x):
    return (1.0 - numpy.exp(-2*x))/(1.0 + numpy.exp(-2*x))


def tanh_derivative(x):
    return (1 + tanh(x))*(1 - tanh(x))


class NeuralNetwork:
    #########
    # Since we are studying the XOR function,
    # for the input layer we need to have two neurons,
    # and for the output layer only one neuron.
    #
    #
    # parameters
    # ----------
    # self:      the class object itself
    # net_arch:  consists of a list of integers, indicating
    #            the number of neurons in each layer, i.e. the network architecture
    #########
    def __init__(self, net_arch):
        # initialized the weights, making sure we also
        # initialize the weights for the biases that we will add later
        self.activity = tanh
        self.activity_derivative = tanh_derivative
        self.layers = len(net_arch)
        self.steps_per_epoch = 1000
        self.arch = net_arch
        self.weights = []

        # range of weight values (-1,1)
        for layer in range(self.layers - 1):
            w = 2 * numpy.random.rand(net_arch[layer] + 1, net_arch[layer + 1]) - 1
            self.weights.append(w)

    #########
    # the fit function will train our network. In the last line,
    # `nn` represents the `NeuralNetwork` class and `predict` is the function in the NeuralNetwork class
    # that we will define later.
    #
    # parameters
    # ----------
    # self:    the class object itself
    # data:    the set of all possible pairs of booleans True or False indicated by the integers 1 or 0
    # labels:  the result of the logical operation 'xor' on each of those input pairs
    #########
    def fit(self, data, labels, learning_rate=0.1, epochs=100):

        errors = list()
        # Add bias units to the input layer -
        # add a "1" to the input data (the always-on bias neuron)
        ones = numpy.ones((1, data.shape[0]))
        Z = numpy.concatenate((ones.T, data), axis=1)

        training = epochs * self.steps_per_epoch
        error_sum = 0
        for k in tqdm(range(training)):
            if k % self.steps_per_epoch == 0:
                '''print('epochs: {}'.format(k/self.steps_per_epoch))
                for s in data:
                    print(s, nn.predict_single_data(s))'''

            # We will now go ahead and set up our feed-forward propagation:
            sample = numpy.random.randint(data.shape[0])
            y = [Z[sample]]
            for i in range(len(self.weights) - 1):
                activation = numpy.dot(y[i], self.weights[i])
                activity = self.activity(activation)

                # add the bias for the next layer
                activity = numpy.concatenate((numpy.ones(1), numpy.array(activity)))
                y.append(activity)

            # last layer
            activation = numpy.dot(y[-1], self.weights[-1])
            activity = self.activity(activation)
            y.append(activity)

            # Now we do our back-propagation of the error to adjust the weights:
            error = labels[sample] - y[-1]

            # errorの記録
            error_sum += abs(error[0])
            if k % self.steps_per_epoch == 0:
                errors.append(error_sum)
                error_sum = 0

            delta_vec = [error * self.activity_derivative(y[-1])]

            # we need to begin from the back, from the next to last layer
            for i in range(self.layers - 2, 0, -1):
                error = delta_vec[-1].dot(self.weights[i][1:].T)
                error = error * self.activity_derivative(y[i][1:])
                delta_vec.append(error)

            # Now we need to set the values from back to front
            delta_vec.reverse()

            # Finally, we adjust the weights, using the backpropagation rules
            for i in range(len(self.weights)):
                layer = y[i].reshape(1, self.arch[i] + 1)
                delta = delta_vec[i].reshape(1, self.arch[i + 1])
                self.weights[i] += learning_rate * layer.T.dot(delta)
        return errors

    #########
    # the predict function is used to check the prediction result of
    # this neural network.
    #
    # parameters
    # ----------
    # self:   the class object itself
    # x:      single input data
    #########
    def predict_single_data(self, x):
        val = numpy.concatenate((numpy.ones(1).T, numpy.array(x)))
        for i in range(0, len(self.weights)):
            val = self.activity(numpy.dot(val, self.weights[i]))
            val = numpy.concatenate((numpy.ones(1).T, numpy.array(val)))
        return val[1]

    #########
    # the predict function is used to check the prediction result of
    # this neural network.
    #
    # parameters
    # ----------
    # self:   the class object itself
    # x:      the input data array
    #########
    def predict(self, X):
        Y = None
        for x in X:
            y = numpy.array([[self.predict_single_data(x)]])
            if Y is None:
                Y = y
            else:
                Y = numpy.vstack((Y, y))
        return Y


def main():
    numpy.random.seed(0)

    # Initialize the NeuralNetwork with
    # 2 input neurons
    # 2 hidden neurons
    # 1 output neuron
    nn = NeuralNetwork([2, 2, 1])
    #nn = NeuralNetwork([2, 4, 3, 1])

    # Set the input data
    X = numpy.array([[0, 0], [0, 1],
                     [1, 0], [1, 1]])

    # Set the labels, the correct results for the xor operation
    y = numpy.array([0, 1,
                     1, 0])

    epochs = [x for x in range(10000)]
    errors = nn.fit(X, y, epochs=10)

    # Show the prediction results
    print("Final prediction")
    for s in X:
        print(s, nn.predict_single_data(s))

    plot_decision_regions(X, y, nn)
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

    plt.xlabel('epochs')
    plt.ylabel('error')
    plt.plot(epochs, errors)
    plt.show()


if __name__ == '__main__':
    main()
