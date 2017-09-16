from importlib import reload
import CNNClass
import csv


def mod(m, n):
    return m - (m // n) * n


answers_list = [[0 for i in range(10)] for j in range(42000)]
image_list = [[0 for j in range(784)] for i in range(42000)]

with open('train.csv') as csvfile:
    reader = csv.reader(csvfile)
    i = 0
    for row in reader:
        if row[0] == 'label':
            continue
        answers_list[i][int(row[0])] = 0.5
        image_list[i] = row[1:]
        i += 1

images_list = []
for s in range(len(image_list)):
    image = [[0 for j in range(28)] for i in range(28)]
    for k in range(len(image_list[s])):
        j = mod(k, 28)
        i = (k - j) // 28
        image[i][j] = int(image_list[s][k]) / 255
    images_list.append(image)

db_name = 'weights.db'
names = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
image_size = 28
filter_size = 5
filters_quantity = 10
pooling_size = 2
hidden_1_size = 600
hidden_2_size = 400
n = 2
cnn = CNNClass.CNNSearch(
    image_size, names, filter_size, filters_quantity, pooling_size, hidden_1_size, hidden_2_size, db_name
)
cnn.training_set(images_list, answers_list, n)




