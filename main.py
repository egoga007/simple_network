import numpy as np
inputs = [  [0, 1, 0, 0],
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 1, 0],
            [1, 1, 1, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 1, 1, 1],
            [1, 0, 0, 1],
            [1, 0, 1, 1],
            [1, 1, 0, 1]]
epochs = 20
print("Начинаю обучение на данных:")
print(inputs)
print("Kолличество эпох обучеия:", epochs)
print("Hажмите Enter для продолжения.")
input()
outputs = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
weights = [0.5, 0.5, 0.5, 0.5]

for e in range(epochs):
    print("Эпоха ", e+1)
    for indx,input in enumerate(inputs):

        sum1 = 0

        for length in range(len(weights)):
            neuron = input[length] * weights[length]
            sum1 += neuron
        sigmoid =1 / (1 + np.exp(-sum1))
        error = outputs[indx] - sigmoid
        sigmoid_derivative=sigmoid*(1-sigmoid)

        # поправка весов
        for l in range(len(weights)):
            weights[l] += input[l] * error * sigmoid_derivative
            print("weight", l, ": ", weights[l])
        print("--------")
print("Обучение закончено." )
print("Измененные веса в результате обучения: ", weights)

print()
#Test
input2 = [[0, 1, 0, 0], [0, 0, 1, 1]]
i = 0

for value in input2:
    sum2 = 0
    print("\nТест нейронной сети на новых данных:", value)
    for l2 in range(len(weights)):
        neuron2 = value[l2] * weights[l2]
        sum2 += neuron2
    i -= 1
    sigmoid2 = 1 / (1 + np.exp(-sum2))

    print("Оценка тестовых данных: ", sigmoid2)
