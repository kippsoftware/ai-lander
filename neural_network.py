"""Working neural network using PyTorch tensors

"""
import random
import torch

def tensor(x = None):
    """Create a PyTorch tensor; random if initializer not provided"""
    x = x if x else random.uniform(-1,1)
    return torch.tensor(x, dtype=torch.float64, requires_grad=True)    

class Neuron:
    """Use tensors to implement y = tanh(wa + b)"""

    def __init__(self, size):
        self.weights = [tensor() for _ in range(size)]
        self.bias = tensor()

    def parameters(self):
        """Report weights and bias to adjust in gradient descent"""
        return self.weights + [self.bias]

    def forward(self, activation):
        """Activate the neuron"""
        return sum((wi * ai for wi, ai in zip(self.weights, activation)), self.bias).tanh()

    def __str__(self, indent = ''):
        weights = ' '.join((f'{wi.item():7.4f}' for wi in self.weights))
        return f"{indent}neuron weights {weights} bias {self.bias:7.4f}"

class Layer:
    """Parallel neurons that each receive the same activation and each produce a result"""

    def __init__(self, activation_size, result_size):
        self.neurons = [Neuron(activation_size) for _ in range(result_size)]

    def parameters(self):
        """Assemble weights and biases to adjust in gradient descent"""
        return [p for neuron in self.neurons for p in neuron.parameters()]

    def forward(self, activation):
        """Activate the neurons"""
        return [neuron.forward(activation) for neuron in self.neurons]

    def __str__(self, indent = ''):
        out = [ f'{indent}layer {len(self.neurons[0].weights)} {len(self.neurons)}' ]
        out.extend(neuron.__str__('  ' + indent) for neuron in self.neurons)
        return '\n'.join(out)

class Perceptron:
    """Multi-layer perceptron (MLP); c.f., PyTorch nn.Sequential"""

    def __init__(self, *sizes):
        self.layers = [Layer(sizes[i], sizes[i+1]) for i in range(len(sizes)-1)]

    def parameters(self):
        """Assemble weights and biases to adjust in gradient descent"""
        return [p for layer in self.layers for p in layer.parameters()]

    def forward(self, activation):
        """Iteratively activate each layer with the output of the previous layer"""
        for layer in self.layers:
            activation = layer.forward(activation)
        return activation

    def loss(self, desired, actual):
        """Compute mean squared error (MSE) to determine loss"""
        return sum((di - ai)**2 for d,a in zip(desired, actual) for di,ai in zip(d,a))

    @torch.no_grad() # avoids PyTorch AssertionError about non-gradient leaf tensors
    def adjust_parameters(self, step_size):
        """Adjust parameters by the step size to reduce loss"""
        for parameter in self.parameters():
            parameter += -step_size if parameter.grad > 0 else step_size

    def train_network(self, training_inputs, desired_outputs, step_size, num_steps, threshold):
        """Activate forward, compute loss, backpropagate gradients, adjust parameters;
        iterate for convergence"""
        for which_step in range(num_steps):
            actual_outputs = [self.forward(activation) for activation in training_inputs]
            loss = self.loss(desired_outputs, actual_outputs)
            print('loss', which_step, loss.item(), threshold)
            if loss.item() < threshold :
                break
            loss.backward()
            self.adjust_parameters(step_size)
        return loss

    def __str__(self):
        sizes = ' '.join([str(len(layer.neurons[0].weights)) for layer in self.layers])
        out = [ f'perceptron {sizes} {len(self.layers[-1].neurons)}' ]
        out.extend(layer.__str__('  ') for layer in self.layers)
        return  '\n'.join(out)

if __name__ == '__main__':
    random.seed(2024)
    neuron = Neuron(2)
    print(neuron)
    activation = [0.0, 1.0]
    print('activation', activation)
    print('forward', neuron.forward(activation).item())

    layer = Layer(2, 2)
    print(layer)
    activation = [0.0, 1.0]
    print('forward', [t.item() for t in layer.forward(activation)])

    perceptron = Perceptron(2, 1)
    print(perceptron)
    training_data = [[ 8.0, -6.0],
                     [-7.0,  0.5],
                     [ 3.0,  0.0],
                     [ 9.0, -9.0]]
    desired_output = [[1.0], [-1.0], [-1.0], [1.0]]
    actual_output = [perceptron.forward(activation) for activation in training_data]
    loss = perceptron.loss(desired_output, actual_output)
    print('loss', loss.item())
    print('parameters', ' '.join(f'{p.item():0.4f}' for p in perceptron.parameters()))
    print(perceptron)

    print('backward')
    loss.backward()
    print('adjust parameters')
    perceptron.adjust_parameters(0.01)
    print(perceptron)
    actual_output = [perceptron.forward(activation) for activation in training_data]
    loss = perceptron.loss(desired_output, actual_output)
    print('loss', loss.item())

    print('train network')
    perceptron.train_network(training_data, desired_output, 0.1, 10, 0.1)
    [print('activate', perceptron.forward(activation)) for activation in training_data]
