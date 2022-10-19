import numpy as np

class CNN:
    def __init__(self):
        #input(9x9x1) -> kernel(3x3x1) -> pool(2x2x1) -> kernel(3x3x1) -> flattened -> activation(RUu) -> neuron -> output -> softmax

        self.input_layer = np.random.randint(5, size=(9,9))
        self.kernals = np.random.randint(5, size=(3,3,2)) #valid
        # self.neurons = []
        # self.flattened_layer = []
        # self.weights = np.random.rand()

    def compute_layer(self,layer_shape, kernel_shape, layer, kernel):
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


    def pool(self,pool_shape,layer_shape,layer):
        #max pooling
        output = []
        for y_pos in range(layer_shape[1] - pool_shape[1] + 1):
            # scoots down the matrix
            output_row = []
            for x_pos in range(layer_shape[0] - pool_shape[0] + 1):
                # scoots across the matrix
                local_max = []
                for matrix_y_level in range(pool_shape[1]):
                    local_max.append(max(layer[matrix_y_level + y_pos][x_pos: x_pos + matrix_y_level]))
                output_row.append(max(local_max))
            output.append(output_row)
        return output








    def softmax(self, output):

        return np.exp(output) / np.sum(np.exp(output), axis=0)

    def activation(self, neuron):
        #tanh activation
        return np.tanh(neuron)






    def softmax(self, output):

        return np.exp(output) / np.sum(np.exp(output), axis=0)

    def activation(self, neuron):
        #tanh activation
        return np.tanh(neuron)
