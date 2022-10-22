import numpy as np


class CNN:
    def __init__(self, *input_matrix, padding=0, strides=0, output_labels=None):
        # During assembly, make layer reshape
        self.output_labels = output_labels
        self.layers = [list(input_matrix)]
        self.kernels = []  # valid
        self.event_stack = []
        if padding != 0:
            self.padding_layer(padding)

        if strides != 0:
            self.stride_layer(strides)

    def add_pool(self, shape):
        self.event_stack.append(f"pool:{shape}")
        return

    def add_kernel(self, amount, shape, max_size: int = 2):
        self.event_stack.append(f"kernel:{shape}")
        self.spawn_kernel(amount, shape, max_size)
        return

    def add_dense(self, neurons, softmax=False, activation=True):
        self.event_stack.append(f"dense:{[neurons, softmax, activation]}")
        return

    def flatten(self):
        self.event_stack.append(f"flat:")
        return

    def spawn_kernel(self, amount, shape, max_size, ):
        local_kernels = []
        for val in range(amount):
            kernel = np.random.randint(max_size, size=(shape[0], shape[1]))
            local_kernels.append(kernel.tolist())
        self.kernels.append(local_kernels)
        return

    def stride_layer(self, stride_size):
        for it in range(stride_size):
            for index in range(len(self.layers[-1])):
                for d1 in range(len(self.layers[-1][index])):
                    self.layers[-1][index][d1].insert(0, 0)
                    self.layers[-1][index][d1].append(0)
                self.layers[-1][index].insert(0, np.zeros(len(self.layers[-1][index]), dtype=np.int8).tolist())
                self.layers[-1][index].append(np.zeros(len(self.layers[-1][index]), dtype=np.int8).tolist())

    def padding_layer(self, padding_layer):
        for it in range(padding_layer):
            for index in range(len(self.layers[-1])):
                for d1 in range(len(self.layers[-1][index])):
                    del self.layers[-1][index][d1][0]
                    del self.layers[-1][index][d1][-1]
                del self.layers[-1][index][0]
                del self.layers[-1][index][-1]

    def run_network(self):
        # kernel [layer] [num/children] [matrix] [1d list]
        # layer [[[]],[[],[],[]],[[],[],[],[],[],[]]]
        # layer [iteration] [child layer] [matrix] [1d list]
        iteration_stack = []
        kernel_layer = -1
        for event in self.event_stack:
            event_name = event.split(":")
            if event_name[0] == "kernel":
                kernel_layer += 1
            iteration_stack = []
            if event_name[0] == "flat":
                self.flatten_layer()
                return
            if event_name[0] == "dense":
                arguments = eval(event_name[1])
                self.dense(arguments[0], softmax=arguments[1], activation=arguments[2])

            for cur_layer in self.layers[-1]:
                if event_name[0] == "kernel":
                    for val in range(len(self.kernels[kernel_layer])):  # picked up
                        compound_layer = self.compute_layer([len(cur_layer[0]), len(cur_layer)], eval(event_name[1]),
                                                            cur_layer, self.kernels[kernel_layer][val])
                        iteration_stack.append(compound_layer)
                elif event_name[0] == "pool":
                    pool_layer = self.pool(eval(event_name[1]), [len(cur_layer[0]), len(cur_layer)], cur_layer)
                    iteration_stack.append(pool_layer)
                else:
                    print("Something went wrong, nice code dummy")
            self.layers.append(iteration_stack)
        return self.layers[-1]

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


    def flatten_layer(self):
        # turns matrix into single list to pass to a neural network
        semi_flat_1 = [item for sublist in self.layers[-1] for item in sublist]
        output = [item for sublist in semi_flat_1 for item in sublist]
        self.layers.append(output)
        return

    def dense(self, neurons, softmax=False, activation=True):
        nn = CNN.NeuralNetwork([len(self.layers[-1]), neurons])
        output = nn.forward_nn(self.layers[-1], softmax=softmax, activation=activation)
        self.layers.append(output)


    def summery(self):
        if self.output_labels != None:
            labels = self.output_labels
            for key in output_lables:
                print(f"{key}: {round((self.layers[-1][labels[key]]*0.1),5)}%")


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

        def forward_nn(self, inputs, softmax=False, activation=True):
            output = []
            self.neurons = []
            self.neurons.append(inputs)
            for layer_pos in range(len(self.network_layers) - 1):  # good?
                neuron_layer = []
                weight_layer_pos = self.weights[layer_pos]
                for neuron_pos in range(self.network_layers[layer_pos + 1]):
                    neuron_layer.append(self.activation(
                        np.dot(self.neurons[layer_pos], [weight[neuron_pos] for weight in weight_layer_pos])))
                self.neurons.append(neuron_layer)
            output.append(self.neurons[-1])

            if softmax == True:
                return self.softmax(output[0])
            else:
                return output[0]

        def softmax(self, output):
            softmax = np.exp(output) / np.sum(np.exp(output), axis=0)
            return softmax.tolist()

        def activation(self, neuron):
            # tanh activation
            return np.tanh(neuron)


# debug stuff
output_lables = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
                 "ten": 10}
layer = np.random.randint(4, size=(9, 9))
layer1 = np.random.randint(4, size=(9, 9))
cnn = CNN(layer.tolist(), layer1.tolist(), output_labels=output_lables)
cnn.add_kernel(10, [4, 4], max_size=5)
cnn.add_pool([2, 2])
cnn.add_kernel(1, [2, 2])
cnn.flatten()
cnn.add_dense(10, softmax=True)
cnn.run_network()
cnn.summery()

