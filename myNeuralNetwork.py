# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 17:43:05 2019

@author: trevi
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 13:52:03 2019

@author: trevi
"""

from sklearn import datasets

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import numpy as np
import math
import matplotlib.pyplot as plt
import random
iris = datasets.load_iris()

normalized_data = preprocessing.normalize(iris.data)


data_train, data_test, targets_train, targets_test = train_test_split(normalized_data ,iris.target , test_size = 0.33, random_state = 0)
trainingIterations = 100 # how many times we want to train the set
x = list(range(0, trainingIterations))# for our graph


# =============================================================================
# # create a node (neuron)
# =============================================================================
class Node:
    def __init__(self, numInputs, has_bias_node = True):
        self.weights = [];
        self.a = 0;
        self.error = 0
        for i in range(0,numInputs + has_bias_node):
            rand = round(random.uniform(-2,2), 3)
#            print("random Number", rand, "number of inputs I:", i)
            self.weights.append(rand)
  
    # calculate output
    def calculate_output(self, inputs):
        theSum = 0;


        for i in range(0, len(inputs)):
            #print("inputs: " + str(len(inputs)))
            #print("weights: " + str(len(self.weights)))
            theSum += inputs[i] * self.weights[i]
        self.a = self.sigmoid(theSum)
        #print('self.a', self.a)
        number = self.a
        return number
      
    def sigmoid(self,x):
        return 1 / (1 + math.exp(-x) )
  


# =============================================================================
# 
# =============================================================================
class Layer:
    def __init__(self, size = 0, numberOfInputs = 0, biasNode = True):
        self.size = size
        self.numberOfInputs = numberOfInputs
        self.nodes = []
        self.hasBias = biasNode

        numberOfInputs += biasNode
        self.generate_nodes()
        
    def generate_nodes(self):
        for i in range(0,self.size):
           self.add_node(Node(self.numberOfInputs, self.hasBias))
           
    def add_node(self, node):
        self.nodes.append(node)
           
       
           
    def get_outputs(self, inputs):
        outputs = []
        
        if self.hasBias:
            inputs = np.append(inputs, -1)
            
            
        for i in range(0, self.size):
            outputs.append(self.nodes[i].calculate_output(inputs))
        return outputs
            



# =============================================================================
# 
# =============================================================================
class Network:
    def __init__(self, lengths_of_layers, num_of_inputs, data = []):
        self.lengths_of_layers = lengths_of_layers
        self.num_of_inputs = num_of_inputs
        self.data = data
        self.layers = self.create_layers(self.lengths_of_layers, self.num_of_inputs)

        
        
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
                else:
                    summ = 0
                    layer_k = self.layers[l]
                    for k in range(0, len(layer_k.nodes)):
                        node_in_k = layer_k.nodes[k]
                        summ += node_in_k.weights[n-1] * node_in_k.error
                        node_in_k.weights[n-1] -= learning_rate * node_in_k.error * node.a
                    error = node.a * (1 - node.a) * summ
                    node.error = error

        for node in self.layers[0].nodes:
            for weight_index in range(0, len(node.weights)-1):
                node.weights[weight_index] -= learning_rate * node.error * inputs[weight_index]

                    
# =============================================================================
# Main is the driver function
# It will build a network
# Train it
# Graph it
# And test it on our test set.                
# =============================================================================
def main():                 
    #Build Network                  
    network = Network([3, 5, 3], len(normalized_data[0]))
    
    
    
    
    predictions = []
    accuracies = []
    #Train Network
    for j in range(0, trainingIterations):
        
        for i in range(0, len(data_train)):
            row = data_train[i]
            target = targets_train[i]
            output = network.calculate_output(row.tolist())
            network.back_propagate(target, row, 1)
            prediction = np.argmax(output)
            predictions.append(prediction)
        #print(predictions)
        acc = accuracy_score(targets_train, predictions)
        print(acc)
        accuracies.append(acc)
        predictions = []
    
    #Plot Training in graph
    plt.plot(x, accuracies)
    plt.show()
    
    
    #Test Network on test data
    predictions = []
    accuracies = []
    for row in data_test:
        output = network.calculate_output(row.tolist())
        prediction = np.argmax(output)
        predictions.append(prediction)
    print("Accuracy score on test data", accuracy_score(targets_test, predictions))                    


if __name__== "__main__":
  main()