# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 10:17:59 2019

@author: trevi
"""

from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

headers = ['buy', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'target']
    
car_input = pd.read_csv("car.data",
                  header=None, names=headers, na_values="?" )

small_car_input = pd.read_csv("small_car.data",
                  header=None, names=headers, na_values="?" )

car_data = car_input[['buy', 'maint', 'doors', 'persons', 'lug_boot', 'safety']]
car_targets = car_input['target']

small_car_data = small_car_input[['buy', 'maint', 'doors', 'persons', 'lug_boot', 'safety']]
small_car_targets = small_car_input['target']
#car_targets = pd.get_dummies(car_targets, columns=['target'])
#car_data = pd.get_dummies(car_data, columns=['buy', 'maint', 'doors', 'persons', 'lug_boot', 'safety'])

#print(small_car_data)
#print(small_car_targets)

#car_targets = car_targets.to_numpy()
#car_data = car_data.to_numpy()

car_data_train, car_data_test, car_targets_train, car_targets_test = train_test_split(car_data, car_targets, test_size=0.30)

loop_counter = 0;
# =============================================================================
# The keep function takes a large chunch of  data and only keeps the data that maches
# the rows we want
# =============================================================================
def keep(value, col, data):
    new_df = pd.DataFrame(columns=['buy', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'target'])
    
    for index, row in data.iterrows():
        global loop_counter 
        loop_counter = loop_counter + 1
        is_a_keeper = False
        #for i in range(len(row)):
            #print(attribute)
            #print(row)
        if value == row[col]:# and i == col:
                #print(attribute)
            is_a_keeper = True
        if is_a_keeper == True:
            new_df = new_df.append(row)
    print("loop_counter", loop_counter)
    return new_df.drop(data.columns[col], axis='columns')

#print("Keep function-------------------------------------------")
#print(keep("vhigh", 1, small_car_input))


# =============================================================================
# returns the entropy
# =============================================================================
from math import log
def entropy(data):
    entropy = 0
    size = len(data)
    counts = data.target.value_counts()
    print("count size", len(counts))
    for count in counts:
        if len(counts) != 1:
            entropy -= (count / size) * log((count / size), 2)
#    print("should print entropy")
#    print(entropy)
    return entropy

#print("Entropy Function cakeells keep once:--------------------------------------")
#print(entropy(keep("small", 4, small_car_input)))


# =============================================================================
# Returns the name of the root and what column number it is
# =============================================================================
def choose_root(data):
    index = 0
    min_index = -1
    min_entropy = 99999999
    print("Choose root Data Length:", len(data), "\n")
    for col_name in data:
        average_entropy = 0.0
        if col_name != "target":
            counts = data[col_name].value_counts()
            labels = counts.keys()
            total_entropy = 0
            print("# times calling Entropy", len(labels), "\n")
            if len(labels) > 0:
                for label in labels:
                    total_entropy += entropy(keep(label, index, data))
                average_entropy = total_entropy / len(labels)
            else:
                average_entropy = 0
           
            if average_entropy != 0:
                print(average_entropy)
            if average_entropy < min_entropy:
                min_index = index
                min_entropy = average_entropy
            index += 1
            print("choose root",data.columns[min_index])
    return data.columns[min_index], min_index

#print("choose Root function------------------------------------------", "\n")
#print(choose_root(car_input))


# =============================================================================
# doesn't do a damb thing
# =============================================================================
class Node:
    def __init__(self, value, children=[]):
        self.value = value
        self.child_nodes = children
       
        

            
# =============================================================================
# Chooses the most common. this is how we choose a leaf wiht a lot of possioble options
# =============================================================================
def most_common(lst):
    return max(set(lst), key=lst.count)

# =============================================================================
#  This is buidling a tree but no in a way i can travers it once it made
# =============================================================================
class Tree:
    def __init__(self, input, level):
        if level != 6:
            print("level", level)
        self.level = level
        self.leaf = False
        self.data = input
        self.nodes = []
        self.root, self.root_index = choose_root(self.data)
        self.target = []
#        self.root = 'safety'
#        self.root_index = 5
        self.build_tree(self.data)
        
    def build_tree(self, data):
        if entropy(data) != 0 and self.level > 0:
            print("*******************************************************")
            counts = data[self.root].value_counts()
            labels = counts.keys()
            for label in labels:
                new_level = (self.level - 1)
                self.nodes.append(Tree(keep(label, self.root_index, data), new_level))
               
        else:
            print("------------------------------------------------------------")
            self.leaf = True
            self.target = self.data.target.value_counts()[:1].index.tolist()
            
print("build Tree function---------------------------------------", "\n")
#tree = Tree(small_car_input, 3)
tree = Tree(car_input, 3)
print(tree.root)

# =============================================================================
# This doesn't work at all
# =============================================================================
predictions = []
sentinal = True
def predict(tree, car_data_test):
    
   print(len(tree.nodes))
   for i in range(len(tree.nodes)):
#       if tree.nodes[i].leaf != False:
#           print(tree.nodes[i].leaf)
#       else:
        for y in range(len(tree.nodes[i].nodes)):
            print("parent", i, "child", y,tree.nodes[i].nodes[y].leaf)
        
#         put the car through the tree till we get to a leaf node and then save target

prediction = predict(tree, car_data_test)
#print(prediction)
print(tree.nodes[1].nodes[1].target)




















