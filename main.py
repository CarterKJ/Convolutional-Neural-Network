import numpy as np


class CNN:
    def __init__(self, input_matrix):
        # goal: input(9x9x1) -> kernel(3x3x1) -> pool(2x2x1) -> kernel(3x3x1) -> flattened -> activation(tanh) -> neuron -> output -> softmax
        self.input = input_matrix
        self.layers = [[input_matrix.tolist()]]
        self.kernels = []  # valid
        self.event_list = []


    def add_pool(self, shape):
        self.event_list.append(f"pool:{shape}")
        return

    def add_kernel(self, amount, shape, max_size: int = 2):
        self.event_list.append(f"kernel:{shape}")
        self.spawn_kernel(amount,shape, max_size)
        return

    def spawn_kernel(self,amount, shape, max_size, ):
        local_kernels=[]
        for val in range(amount):
            kernel = np.random.randint(max_size, size=(shape[0], shape[1]))
            local_kernels.append(kernel.tolist())
        self.kernels.append(local_kernels)
        return


    def forward(self):
        # kernal [layer] [num/chidren] [matrix] [1d list]
        # layer [[[]],[[],[],[]],[[],[],[],[],[],[]]]
        # layer [iteration] [child layer] [matrix] [1d list]
        iteration_stack = []
        kernel_layer = -1
        for event in self.event_list:
            event_name = event.split(":")
            if event_name[0] == "kernel":
                kernel_layer += 1
            iteration_stack = []
            for cur_layer in self.layers[-1]:
                if event_name[0] == "kernel":
                    for val in range(len(self.kernels[kernel_layer])): #left off
                        # for matrix in cur_layer:
                        compound_layer = self.compute_layer([len(cur_layer[0]), len(cur_layer)], eval(event_name[1]), cur_layer, self.kernels[kernel_layer][val])
                        iteration_stack.append(compound_layer)
                elif event_name[0] == "pool":
                    pool_layer = self.pool(eval(event_name[1]), [len(cur_layer[0]), len(cur_layer)], cur_layer)
                    # debug.append(matrix)
                    iteration_stack.append(pool_layer)
                else:
                    print("Something went wrong, nice code dummy")
            self.layers.append(iteration_stack)
        return self.layers


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
        semi_flat_1 = [item for sublist in layer for item in sublist]
        semi_flat_2 = [item for sublist in semi_flat_1 for item in sublist]
        output = [item for sublist in semi_flat_2 for item in sublist]


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
        for layer_pos in range(len(self.network_layers) - 1):  # good?
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

# debug stuff
layer = np.random.randint(4, size=(9, 9))
cnn = CNN(layer)
cnn.add_kernel(10,[2,2], max_size=5)
cnn.add_pool([2,2])
cnn.add_kernel(3,[2,2])
cn_out = cnn.flatten(cnn.forward())
nn = NeuralNetwork([len(cn_out), 50, 20])
print(nn.forward(cn_out))
