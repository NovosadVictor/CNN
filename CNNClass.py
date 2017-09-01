from random import random
from math import exp, tanh



def sigmoid(y):
    return tanh(y)


def d_sigmoid(y):
    return 1 - (y * y)


def mod(k, n):
    return k - (k // n) * n


class CNNSearch():

    def __init__(self, image_size, names, filter_size, pooling_size, hidden_size):
        self.setup_network(image_size, names, filter_size, pooling_size, hidden_size)

    def setup_image(self, image):
        image = image.copy()
        for i in range(len(image)):
            for j in range(len(image[0])):
                image[i][j] = sigmoid(image[i][j])
        self.input = image

    def setup_network(self, image_size, names, filter_size, pooling_size, hidden_size):
        self.names = names

        self.filters = [
            [[(2 * random()) - 1  for j in range(filter_size)] for i in range(filter_size)],
            [[(2 * random()) - 1 for j in range(filter_size)] for i in range(filter_size)],
            [[(2 * random()) - 1 for j in range(filter_size)] for i in range(filter_size)],
            [[(2 * random()) - 1 for j in range(filter_size)] for i in range(filter_size)],
            [[(2 * random()) - 1 for j in range(filter_size)] for i in range(filter_size)],
            [[(2 * random()) - 1 for j in range(filter_size)] for i in range(filter_size)],
            [[(2 * random()) - 1 for j in range(filter_size)] for i in range(filter_size)],
            [[(2 * random()) - 1 for j in range(filter_size)] for i in range(filter_size)],
            [[(2 * random()) - 1 for j in range(filter_size)] for i in range(filter_size)],
            [[(2 * random()) - 1 for j in range(filter_size)] for i in range(filter_size)]
            ]

        self.pooling_size = pooling_size
        fully_input_size = (((image_size - filter_size + 1) * (image_size - filter_size + 1)) / (pooling_size * pooling_size)) * len(self.filters)

        self.fully_connected = [0 for i in range(3)]
        self.fully_connected[0] = []
        self.fully_connected[1] = [0 for i in range(hidden_size)]
        self.fully_connected[2] = [0 for i in range(len(self.names))]

        self.ih_weights = [[(2 * random()) - 1 for j in range(int(fully_input_size))]
                           for i in range(len(self.fully_connected[1]))]
        self.ho_weights = [[(2 * random()) - 1 for j in range(len(self.fully_connected[1]))]
                           for i in range(len(self.fully_connected[2]))]

    def multiplication(self, i, j, filter):
        return sum([
            self.input[i + k][j + l] * self.filters[filter][k][l]
            for l in range(len(self.filters[filter][0]))
            for k in range(len(self.filters[filter]))
        ])

    def maximum(self, i, j, layer):
        return max([
            self.convolution_layers[layer][i][j],
            self.convolution_layers[layer][i + 1][j],
            self.convolution_layers[layer][i][j + 1],
            self.convolution_layers[layer][i + 1][j + 1]
        ])

    def feed_forward(self):
        convolution_layers = [
            [[0 for j in range(len(self.input) - len(self.filters[0]) + 1)]
             for i in range(len(self.input) - len(self.filters[0]) + 1)]
            for s in range(len(self.filters))
        ]
        self.fully_connected[0] = []

        for s in range(len(convolution_layers)):
            for i in range(len(convolution_layers[s])):
                for j in range(len(convolution_layers[s][0])):
                    convolution_layers[s][i][j] = sigmoid(self.multiplication(i, j, s))

        self.convolution_layers = convolution_layers

        pooling_layers = [[[0 for j in range(int(len(self.convolution_layers[s][0]) / self.pooling_size))]
                           for i in range(int(len(self.convolution_layers[s]) / self.pooling_size))]
                          for s in range(len(self.convolution_layers))]

        for s in range(len(pooling_layers)):
            for i in range(0, len(self.convolution_layers[s]), self.pooling_size):
                for j in range(0, len(self.convolution_layers[s][0]), self.pooling_size):
                    pooling_layers[s][i // 2][j // 2] = self.maximum(i, j, s)

        self.pooling_layers = pooling_layers

        for s in range(len(self.pooling_layers)):
            for i in range(len(self.pooling_layers[s])):
                for j in range(len(self.pooling_layers[s][0])):
                    self.fully_connected[0].append(self.pooling_layers[s][i][j])

        for i in range(len(self.fully_connected[1])):
            for j in range(len(self.fully_connected[0])):
                self.fully_connected[1][i] += self.fully_connected[0][j] * self.ih_weights[i][j]
            self.fully_connected[1][i] = sigmoid(self.fully_connected[1][i])

        for i in range(len(self.fully_connected[2])):
            for j in range(len(self.fully_connected[1])):
                self.fully_connected[2][i] += self.fully_connected[1][j] * self.ho_weights[i][j]
            self.fully_connected[2][i] = sigmoid(self.fully_connected[2][i])

    def error(self, answers):
        return sum([
            (self.fully_connected[2][i] - answers[i]) * (self.fully_connected[2][i] - answers[i])
            for i in range(len(self.fully_connected[2])) / 2
        ])

    def d_error(self, i, answer):
        return answer - self.fully_connected[2][i]

    def back_propagation(self, learning_coefficient, answers):
        output_deltas = [0 for i in range(len(self.fully_connected[2]))]
        for i in range(len(output_deltas)):
            output_deltas[i] = self.d_error(i, answers[i]) * d_sigmoid(self.fully_connected[2][i])

        hidden_deltas = [0 for i in range(len(self.fully_connected[1]))]
        for i in range(len(hidden_deltas)):
            for j in range(len(output_deltas)):
                hidden_deltas[i] += output_deltas[j] * self.ho_weights[j][i] * d_sigmoid(self.fully_connected[1][i])

        input_d = [0 for i in range(len(self.fully_connected[0]))]
        for i in range(len(input_d)):
            for j in range(len(hidden_deltas)):
                input_d[i] += hidden_deltas[j] * self.ih_weights[j][i]

        pooling_d = [
            [[0 for j in range(len(self.pooling_layers[s][0]))]
             for i in range(len(self.pooling_layers[s]))]
            for s in range(len(self.pooling_layers))
            ]
        s = 0
        for k in range(len(input_d)):
            # k = j + i * n + s * n * n
            if k > 0 and k % (len(self.pooling_layers[s]) * len(self.pooling_layers[s][0])) == 0:
                s += 1
            j = mod(k, len(self.pooling_layers[s]))
            i = int((k - j - s * len(self.pooling_layers[s]) * len(self.pooling_layers[s])) / len(self.pooling_layers[s]))
            pooling_d[s][i][j] = input_d[k]

        convolution_d = [
            [[0 for j in range(len(self.convolution_layers[s][0]))]
             for i in range(len(self.convolution_layers[s]))]
            for s in range(len(self.convolution_layers))
            ]
        for s in range(len(self.pooling_layers)):
            for i in range(len(self.pooling_layers[s])):
                for j in range(len(self.pooling_layers[s][0])):
                    for l in range(self.pooling_size):
                        for m in range(self.pooling_size):
                            convolution_d[s][i + l][j + m] = pooling_d[s][i][j]

        filters_deltas = [[[0 for j in range(len(self.filters[s][0]))]
                           for i in range(len(self.filters[s]))]
                          for s in range(len(self.filters))]
        for s in range(len(filters_deltas)):
            for i in range(len(filters_deltas[s])):
                for j in range(len(filters_deltas[s][0])):
                    filters_deltas[s][i][j] = convolution_d[s][i][j] * d_sigmoid(self.convolution_layers[s][i][j])

        for i in range(len(self.fully_connected[2])):
            for j in range(len(self.fully_connected[1])):
                change = output_deltas[i] * self.fully_connected[1][j]
                self.ho_weights[i][j] += learning_coefficient * change

        for i in range(len(self.fully_connected[1])):
            for j in range(len(self.fully_connected[0])):
                change = hidden_deltas[i] * self.fully_connected[0][j]
                self.ih_weights[i][j] += learning_coefficient * change

        for s in range(len(self.filters)):
            for i in range(len(self.filters[s])):
                for j in range(len(self.filters[s][0])):
                    change = 0
                    for l in range(len(self.convolution_layers[s])):
                        for m in range(len(self.convolution_layers[s][0])):
                            change += convolution_d[s][l][m] * self.input[i + l][j + m]
                    self.filters[s][i][j] += learning_coefficient * change

    def training(self, image, answers):
        self.setup_image(image)
        self.feed_forward()
        self.back_propagation(learning_coefficient=0.4, answers=answers)

    def training_set(self, image_list, answers_list, n):
        for j in range(n):
            for i in range(len(image_list)):
                self.training(image_list[i], answers_list[i])

    def get_result(self, image):
        self.setup_image(image)
        self.feed_forward()
        return self.fully_connected[2][:]
