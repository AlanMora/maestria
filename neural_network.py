"""Python port of NeuralNetwork / Perceptrons / RBF (pure Python, no dependencies)."""
import math
import random


def _rand_weight(inputs):
    # Xavier-style init: big enough to break symmetry, scaled by fan-in
    limit = 1.0 / math.sqrt(inputs)
    return random.uniform(-limit, limit)


class Perceptron:
    def __init__(self, inputs):
        self.n = inputs
        # last weight is the homogeneous (bias) coordinate
        self.w = [_rand_weight(inputs) for _ in range(inputs + 1)]
        self.R = [0.0] * inputs
        self.r = 0.5  # learning rate
        self.sig = 1.0
        self.out = 0.0

    def evaluate(self, x):
        s = self.w[self.n]
        for i in range(self.n):
            s += self.w[i] * x[i]
        self.out = 1.0 / (1.0 + math.exp(-s / self.sig))
        return self.out

    def clear(self):
        self.R = [0.0] * self.n

    def training(self, x, E):
        # derivative of 1/(1+exp(-s/sig)) is out*(1-out)/sig
        g = E * self.out * (1.0 - self.out) / self.sig
        for i in range(self.n):
            self.R[i] += g * self.w[i]  # uses the weight before updating it
            self.w[i] += self.r * g * x[i]
        self.w[self.n] += self.r * g
        return self.out


class RBF(Perceptron):
    def evaluate(self, x):
        s = 0.0
        for i in range(self.n):
            d = x[i] - self.w[i]
            s -= d * d
        self.out = math.exp(s)
        return self.out

    def training(self, x, E):
        g = E * self.out * (1.0 - self.out)
        for i in range(self.n):
            # d out / d x[i] = 2 * out * (w[i] - x[i])
            self.R[i] += E * 2.0 * self.out * (self.w[i] - x[i])
            self.w[i] += self.r * g * self.out * (x[i] - self.w[i])
        self.w[self.n] += self.r * g
        return self.out


class InputLayer:
    def __init__(self, inputs):
        self.outputs = [0.0] * inputs
        self.prev = None
        self.next = None

    def set_input(self, data):
        self.outputs = list(data)

    def forward(self):
        pass

    def backward(self):
        pass


class NeuronLayer:
    """Fully connected layer of Perceptron or RBF neurons."""
    def __init__(self, prev, n_neurons, neuron_cls=Perceptron):
        self.prev = prev
        self.next = None
        prev.next = self
        n_inputs = len(prev.outputs)
        self.neurons = [neuron_cls(n_inputs) for _ in range(n_neurons)]
        self.outputs = [0.0] * n_neurons
        self.back_deltas = [[] for _ in range(n_neurons)]

    def forward(self):
        x = self.prev.outputs
        self.outputs = [nr.evaluate(x) for nr in self.neurons]

    def backward(self):
        x = self.prev.outputs
        nxt = self.next
        for i, nr in enumerate(self.neurons):
            nr.clear()
            # total error reaching neuron i = sum of what every next neuron sends back
            E = sum(nxt.back_deltas[k][i] for k in range(len(nxt.back_deltas)))
            nr.training(x, E)
            self.back_deltas[i] = nr.R


class OutputLayer:
    """Picks the winner on forward pass and computes error deltas on backward pass."""
    def __init__(self, prev):
        self.prev = prev
        self.next = None
        prev.next = self
        n = len(prev.outputs)
        self.outputs = [0.0] * n  # desired outputs
        self.back_deltas = [[0.0] * n for _ in range(n)]
        self.best = -1
        self.confidence = 0.0
        self.error = 0.0

    def set_input(self, data):
        self.outputs = list(data)

    def forward(self):
        best, mx = -1, 0.0
        for i, v in enumerate(self.prev.outputs):
            if v > mx:
                mx, best = v, i
        self.best, self.confidence = best, mx

    def backward(self):
        n = len(self.outputs)
        self.error = 0.0
        for i in range(n):
            row = [0.0] * n
            E = self.outputs[i] - self.prev.outputs[i]
            row[i] = E
            self.error += abs(E)
            self.back_deltas[i] = row


class NeuralNetwork:
    def __init__(self):
        self.layers = []

    def add_input_layer(self, inputs):
        self.layers = [InputLayer(inputs)]

    def add_fully_connected_layer(self, neurons):
        self.layers.append(NeuronLayer(self.layers[-1], neurons, Perceptron))

    def add_rbf_layer(self, neurons):
        self.layers.append(NeuronLayer(self.layers[-1], neurons, RBF))

    def add_output_layer(self):
        self.layers.append(OutputLayer(self.layers[-1]))

    def forward(self):
        for layer in self.layers[1:]:
            layer.forward()

    def backward(self):
        for layer in reversed(self.layers[1:]):
            layer.backward()

    def train(self, inputs, desired):
        self.layers[0].set_input(inputs)
        self.layers[-1].set_input(desired)
        self.forward()
        self.backward()
        return self.layers[-1].error

    def evaluate(self, inputs):
        """Returns (selected_class, confidence)."""
        self.layers[0].set_input(inputs)
        self.forward()
        out = self.layers[-1]
        return out.best, out.confidence


def build_network():
    nn = NeuralNetwork()
    nn.add_input_layer(2)
    nn.add_fully_connected_layer(10)
    nn.add_fully_connected_layer(10)
    #nn.add_rbf_layer(15)
    nn.add_fully_connected_layer(5)
    nn.add_output_layer()
    return nn
