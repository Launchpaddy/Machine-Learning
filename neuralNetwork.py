# -*- coding: utf-8 -*-
"""

@author: trevi
"""

from sklearn import datasets
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

iris = datasets.load_iris()
normalized_iris_data = preprocessing.normalize(iris.data)
#print(len(normalized_iris_data[0]))
data_train, data_test, targets_train, targets_test = train_test_split(normalized_iris_data, iris.target, test_size = 0.33, random_state = 0)



from random import random
import math

class Node:
    def __init__(self, how_many_inputs, has_bias_node = True):
        self.weights = []
        self.a = 0
        self.error = 0
        #print("rando")
        for i in range(how_many_inputs + has_bias_node):
            rando = 1 * (random() - 0.5)
            #print(rando)
            self.weights.append(rando)
        
    def calculate_output(self, inputs):
        sum = 0
        for i in range(0, len(inputs)):
            #print("inputs: " + str(len(inputs)))
            #print("weights: " + str(len(self.weights)))
            sum += inputs[i] * self.weights[i]
        self.a = self.sigmoid(sum)
        return self.a
        
    def sigmoid(self, x):
        return 1.0 / (1 + math.exp(-x))
        
n = Node(4)
n.calculate_output([5, 9, -76, -5])

class Layer:
    def __init__(self, size = 0, num_of_inputs = 0, has_bias_node = True):
        self.size = size
        self.num_of_inputs = num_of_inputs
        self.has_bias_node = has_bias_node
        self.nodes = []
        self.generate_nodes()
        
        
    def generate_nodes(self):
        for i in range(0, self.size):
            #print(self.num_of_inputs)
            #print(self.has_bias_node)
            self.add_node(Node(self.num_of_inputs, self.has_bias_node))
            
    def add_node(self, node):
        self.nodes.append(node)
        
    def get_outputs(self, inputs):
        outputs = []
        if self.has_bias_node:
            inputs.append(-1)
        for i in range(0, self.size):
            outputs.append(self.nodes[i].calculate_output(inputs))
        return outputs
        
l = Layer(3, 2)
l.get_outputs([9, -7])

class Network:
    def __init__(self, lengths_of_layers, num_of_inputs, data = []):
        self.lengths_of_layers = lengths_of_layers
        self.num_of_inputs = num_of_inputs
        self.data = data
        self.layers = self.create_layers(self.lengths_of_layers, self.num_of_inputs)
        #print(self.layers[0])
        
        
    def create_layers(self, lengths, num_of_inputs):
        num_of_prev_nodes = num_of_inputs
        layers = []
        for i in range(0, len(lengths)):
            layers.append(Layer(lengths[i], num_of_prev_nodes))
            num_of_prev_nodes = lengths[i]
        return layers
            
    def calculate_output(self, row):
        current_inputs = row
        for layer in self.layers:
            current_inputs = layer.get_outputs(current_inputs)
        return current_inputs

    def back_propagate(self, target, inputs, learning_rate = 0.3):
        for l in range(len(self.layers), 0, -1):
            layer = self.layers[l-1]
            for n in range(len(layer.nodes), 0, -1):
                node = layer.nodes[n-1]
                if l == len(self.layers):
                    if (n-1) == target:
                        t = 1
                    else:
                        t = 0
                    error = node.a * (1 - node.a) * (node.a - t)
                    node.error = error
                    #print(error)
                else:
                    summ = 0
                    layer_k = self.layers[l]
                    for k in range(0, len(layer_k.nodes)):
                        node_in_k = layer_k.nodes[k]
                        summ += node_in_k.weights[n-1] * node_in_k.error
                        #print("weight " + str(k) + ": " + str(node_in_k.weights[n-1]))
                        node_in_k.weights[n-1] -= learning_rate * node_in_k.error * node.a
                        #print("weight " + str(k) + ": " + str(node_in_k.weights[n-1]))
                    error = node.a * (1 - node.a) * summ
                    node.error = error
                    #print(error)
        for node in self.layers[0].nodes:
            #node.weight
            for weight_index in range(0, len(node.weights)-1):
                node.weights[weight_index] -= learning_rate * node.error * inputs[weight_index]
                    #for
                    #error = 
                
        
        
        
n = Network([3, 5, 3], len(normalized_iris_data[0]))
#print(n.calculate_output([5, -9, 6]))
import matplotlib.pyplot as plt
size = 100
x = list(range(0, size))

layer = Layer(3, len(normalized_iris_data[0]))
predictions = []
accuracies = []
for j in range(0, size):
    for i in range(0, len(data_train)):
        row = data_train[i]
        target = targets_train[i]
        output = n.calculate_output(row.tolist())
        n.back_propagate(target, row, 1)
        prediction = np.argmax(output)
        predictions.append(prediction)
    #print(predictions)
    acc = accuracy_score(targets_train, predictions)
    print(acc)
    accuracies.append(acc)
    predictions = []
#print(predictions)
#for row in normalized_iris_data:
plt.plot(x, accuracies)
plt.show()

predictions = []
accuracies = []
for row in data_test:
    output = n.calculate_output(row.tolist())
    prediction = np.argmax(output)
    predictions.append(prediction)
print(accuracy_score(targets_test, predictions))
