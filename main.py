import numpy as np


class CNN:
    def __init__(self, input_matrix):
        # goal: input(9x9x1) -> kernel(3x3x1) -> pool(2x2x1) -> kernel(3x3x1) -> flattened -> activation(tanh) -> neuron -> output -> softmax
        self.input = input_matrix
        self.layer = input_matrix
        self.kernels = []  # valid
        self.event_list = []

    def add_pool(self, shape):
        self.event_list.append(f"pool:{shape}")
        return

    def add_kernel(self, shape, max_size: int = 2):
        self.event_list.append(f"kernel:{shape}")
        self.spawn_kernel(shape, max_size)
        return

    def spawn_kernel(self, shape, max_size):
        kernel = np.random.randint(max_size, size=(shape[0], shape[1]))
        self.kernels.append(kernel.tolist())
        return


    def forward(self):
        kernel_layer = 0
        for event in self.event_list:
            event = event.split(":")
            if event[0] == "kernel":
                self.layer = self.compute_layer([len(self.layer[0]), len(self.layer)], eval(event[1]), self.layer, self.kernels[kernel_layer])
                kernel_layer += 1
            elif event[0] == "pool":
                self.layer = self.pool(eval(event[1]),[len(self.layer[0]), len(self.layer)], self.layer)
            else:
                print("Something went wrong, nice code dummy")
        return self.layer

    def compute_layer(self, layer_shape, kernel_shape, layer, kernel):
        # 2d valid
        output = []
        for y_pos in range(layer_shape[1] - kernel_shape[1] + 1):
            # scoots down the matrix
            output_row = []
            for x_pos in range(layer_shape[0] - kernel_shape[0] + 1):
                # scoots across the matrix
                output_index = 0
                for kernel_y_level in range(kernel_shape[1]):
                    # splits from into individual lists to sum together to get 1 output value
                    output_index += np.dot(layer[y_pos + kernel_y_level][x_pos: x_pos + kernel_shape[1]],
                                           kernel[kernel_y_level])
                output_row.append(output_index)
            output.append(output_row)
        return output

    def pool(self, pool_shape, layer_shape, layer):
        # max pooling
        output = []
        for y_pos in range(0, layer_shape[1] - pool_shape[1] + 1, pool_shape[1]):
            # scoots down the matrix
            output_row = []
            for x_pos in range(0, layer_shape[0] - pool_shape[0] + 1, pool_shape[0]):
                # scoots across the matrix
                local_max = []
                for matrix_y_level in range(pool_shape[1]):
                    # slits matrix to size of pool, then splits horizontal
                    local_max.append(max(layer[matrix_y_level + y_pos][x_pos: x_pos + pool_shape[0]]))
                # finds max of horizontal splits
                output_row.append(max(local_max))
            output.append(output_row)
        return output

    def flatten(self, layer):
        output = []
        # truns matrix into single list to pass to a neural network
        for val in layer: output += val
        return output


class NeuralNetwork:
    def __init__(self, network_shape):
        self.weights = []
        self.biases = []
        self.neurons = []
        self.network_layers = network_shape
        for layer_i in range(len(network_shape) - 1):
            layer_weight = 0.01 * np.random.randn(network_shape[layer_i], network_shape[layer_i + 1])
            layer_bias = np.zeros((1, network_shape[layer_i + 1]))
            self.weights += [layer_weight.tolist()]
            self.biases += layer_bias.tolist()

    def forward(self, inputs):
        output = []
        self.neurons = []
        self.neurons.append(inputs)
        for layer_pos in range(len(self.network_layers) - 1):  # good
            neuron_layer = []
            weight_layer_pos = self.weights[layer_pos]
            for neuron_pos in range(self.network_layers[layer_pos + 1]):
                neuron_layer.append(self.activation(
                    np.dot(self.neurons[layer_pos], [weight[neuron_pos] for weight in weight_layer_pos])))
            self.neurons.append(neuron_layer)
        output.append(self.neurons[-1])
        return self.softmax(output[0])

    def softmax(self, output):
        softmax = np.exp(output) / np.sum(np.exp(output), axis=0)
        return softmax.tolist()

    def activation(self, neuron):
        # tanh activation
        return np.tanh(neuron)
