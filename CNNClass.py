from random import random
from math import exp


def sigmoid(y):
    return (1 / (1 + exp(-y))) - 0.5


def d_sigmoid(y):
    return (y + 0.5) * (0.5 - y)


def mod(k, n):
    return k - (k // n) * n


class CNNSearch():

    def __init__(
            self, image_size, names, filter_size, filters_quantity, pooling_size, hidden_1_size, hidden_2_size
    ):
        self.setup_network(
            image_size, names, filter_size, filters_quantity, pooling_size, hidden_1_size, hidden_2_size
        )

    def setup_image(self, image):
        self.input = image.copy()

    def setup_network(
            self, image_size, names, filter_size, filters_quantity, pooling_size, hidden_1_size, hidden_2_size
    ):
        self.names = names

        self.filters = [
            [[(2 * random()) - 1 for j in range(filter_size)] for i in range(filter_size)]
            for s in range(filters_quantity)
            ]

        self.pooling_size = pooling_size
        fully_input_size =\
            (((image_size - filter_size + 1) * (image_size - filter_size + 1)) / (pooling_size * pooling_size)) *\
            filters_quantity

        self.fully_connected = [[] for i in range(4)]
        self.fully_connected[1] = [0 for i in range(hidden_1_size)]
        self.fully_connected[2] = [0 for i in range(hidden_2_size)]
        self.fully_connected[3] = [0 for i in range(len(self.names))]

        self.ih_weights = [[(2 * random()) - 1 for j in range(hidden_1_size)]
                           for i in range(fully_input_size)]
        self.hh_weights = [[(2 * random()) - 1 for j in range(hidden_2_size)]
                           for i in range(hidden_1_size)]
        self.ho_weights = [[(2 * random()) - 1 for j in range(len(self.names))]
                           for i in range(hidden_2_size)]

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
            [[0.0 for j in range(len(self.input) - len(self.filters[0]) + 1)]
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

        for j in range(len(self.fully_connected[1])):
            for i in range(len(self.fully_connected[0])):
                self.fully_connected[1][j] += self.fully_connected[0][i] * self.ih_weights[i][j]
            self.fully_connected[1][j] = sigmoid(self.fully_connected[1][j])

        for j in range(len(self.fully_connected[2])):
            for i in range(len(self.fully_connected[1])):
                self.fully_connected[2][j] += self.fully_connected[1][i] * self.hh_weights[i][j]
            self.fully_connected[2][j] = sigmoid(self.fully_connected[2][j])
        for j in range(len(self.fully_connected[3])):
            for i in range(len(self.fully_connected[2])):
                self.fully_connected[3][j] += self.fully_connected[2][i] * self.ho_weights[i][j]
            self.fully_connected[3][j] = sigmoid(self.fully_connected[3][j])

    def error(self, answers):
        return sum([
            (self.fully_connected[3][i] - answers[i]) * (self.fully_connected[3][i] - answers[i])
            for i in range(len(self.fully_connected[3]))
        ]) / 2

    def d_error(self, i, answer):
        return answer - self.fully_connected[3][i]

    def back_propagation(self, learning_coefficient, answers):
        output_deltas = [0 for i in range(len(self.fully_connected[3]))]
        for i in range(len(output_deltas)):
            output_deltas[i] = self.d_error(i, answers[i]) * d_sigmoid(self.fully_connected[3][i])

        hidden_2_deltas = [0 for i in range(len(self.fully_connected[2]))]
        for i in range(len(hidden_2_deltas)):
            for j in range(len(output_deltas)):
                hidden_2_deltas[i] += output_deltas[j] * self.ho_weights[i][j] * d_sigmoid(self.fully_connected[2][i])

        hidden_1_deltas = [0 for i in range(len(self.fully_connected[1]))]
        for i in range(len(hidden_1_deltas)):
            for j in range(len(hidden_2_deltas)):
                hidden_1_deltas[i] += hidden_2_deltas[j] * self.hh_weights[i][j] * d_sigmoid(self.fully_connected[1][i])

        input_d = [0 for i in range(len(self.fully_connected[0]))]
        for i in range(len(input_d)):
            for j in range(len(hidden_1_deltas)):
                input_d[i] += hidden_1_deltas[j] * self.ih_weights[i][j]

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
            i = int(
                (k - j - s * len(self.pooling_layers[s]) * len(self.pooling_layers[s])) / len(self.pooling_layers[s])
            )
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

        filters_deltas = [[[0.0 for j in range(len(self.filters[s][0]))]
                           for i in range(len(self.filters[s]))]
                          for s in range(len(self.filters))]
        for s in range(len(filters_deltas)):
            for i in range(len(filters_deltas[s])):
                for j in range(len(filters_deltas[s][0])):
                    filters_deltas[s][i][j] = convolution_d[s][i][j] * d_sigmoid(self.convolution_layers[s][i][j])

        # update weights and filters
        for j in range(len(self.fully_connected[3])):
            for i in range(len(self.fully_connected[2])):
                change = output_deltas[j] * self.fully_connected[2][i]
                self.ho_weights[i][j] += learning_coefficient * change

        for j in range(len(self.fully_connected[2])):
            for i in range(len(self.fully_connected[1])):
                change = hidden_2_deltas[j] * self.fully_connected[1][i]
                self.hh_weights[i][j] += learning_coefficient * change

        for j in range(len(self.fully_connected[1])):
            for i in range(len(self.fully_connected[0])):
                change = hidden_1_deltas[j] * self.fully_connected[0][i]
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
        self.back_propagation(learning_coefficient=0.5, answers=answers)
        print(self.error(answers))

    def training_set(self, image_list, answers_list, n):
        for j in range(n):
            for i in range(len(image_list)):
                print(i, n)
                self.training(image_list[i], answers_list[i])

    def get_result(self, image):
        self.setup_image(image)
        self.feed_forward()
        return self.fully_connected[2][:]
