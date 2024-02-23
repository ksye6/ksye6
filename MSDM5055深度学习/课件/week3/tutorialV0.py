import numpy as np

def finiteDiff(fn, x, parameters, delta, multiplier=1):
    shape = parameters.shape
    var = parameters.reshape(-1)
    diff = []
    for idx in range(len(var)):
        var[idx] += delta/2
        yplus = fn(x)
        var[idx] -= delta
        yminus = fn(x)
        varDiff = ((yplus - yminus) / delta * multiplier).sum()
        var[idx] += delta / 2

        diff.append(varDiff)

    return np.array(diff).reshape(*shape)


class Node:
    def __init__(self, name, parameters=None):
        self.name = name
        self.parameters = parameters
        self.parameters_deltas = [None for _ in range(len(self.parameters))]



class Linear(Node):
    def __init__(self, input_shape, output_shape, weight=None, bias=None):
        if weight is None:
            weight = np.random.randn(input_shape, output_shape) * 0.01
        if bias is None:
            bias = np.zeros(output_shape)
        super(Linear, self).__init__('linear', [weight, bias])

    def forward(self, x):
        self.x = x
        return np.matmul(x, self.parameters[0]) + self.parameters[1]

    def backward(self, delta):
        self.parameters_deltas[0] = self.x.T.dot(delta)
        self.parameters_deltas[1] = np.sum(delta, 0)
        return delta.dot(self.parameters[0].T)


class Sigmoid(Node):
    def __init__(self):
        super(Sigmoid, self).__init__('sigmoid', [])

    def forward(self, x):
        self.x = x
        self.y = 1.0 / (1.0 + np.exp(-x))
        return self.y

    def backward(self, delta):
        return delta * ((1 - self.y) * self.y)


class Mean(Node):
    def __init__(self):
        super(Mean, self).__init__('mean', [])

    def forward(self, x):
        self.x = x
        return x.mean()

    def backward(self, detla):
        return detla * np.ones(self.x.shape) / np.prod(self.x.shape)


class MSE(Node):
    def __init__(self):
        super(MSE, self).__init__('MSE', [])

    def forward(self, x, l):
        self.l = l
        self.x = x
        return (x - l) ** 2

    def backward(self, delta):
        return 2 * (self.x - self.l) * delta

def net_forward(net, x, label):
    for node in net:
        if node.name == "MSE":
            result = x
            x = node.forward(x, label)
        else:
            x = node.forward(x)
    return result, x

def net_backward(net):
    y_delta = 1.0
    for node in net[::-1]:
        y_delta = node.backward(y_delta)
    return y_delta


if __name__ == "__main__":
    delta = 0.01
    precision = 1e-3

    weight = np.random.randn(5, 10) * 0.01
    bias = np.random.randn(10) * 0.01
    linear = Linear(10, 5, weight, bias)

    x = np.random.uniform(0.1, 1, [6, 5])

    weightFD = finiteDiff(linear.forward, x, weight, delta)
    biasFD = finiteDiff(linear.forward, x, bias, delta)
    xFD = finiteDiff(linear.forward, x, x, delta)

    y = linear.forward(x)
    xDiff = linear.backward(np.ones([6, 10]))

    sigmoid = Sigmoid()

    xFD = finiteDiff(sigmoid.forward, x, x, delta)

    y = sigmoid.forward(x)
    xDiff = sigmoid.backward(np.ones([6, 5]))

    graph = [linear, sigmoid]

    weightFD = finiteDiff(lambda x: net_forward(graph, x), x, weight, delta)
    biasFD = finiteDiff(lambda x: net_forward(graph, x), x, bias, delta)
    xFD = finiteDiff(lambda x: net_forward(graph, x), x, x, delta)

    y = net_forward(graph, x)
    xDiff = net_backward(graph, np.ones([6, 10]))

    (xDiff - xFD)
    (weightFD-linear.parameters_deltas[0])
    (biasFD-linear.parameters_deltas[1])
