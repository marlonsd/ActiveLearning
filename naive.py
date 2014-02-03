#!/usr/bin/python

import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.datasets import *
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt # Ploting

repetition = 5 # The algorithm will run this number of times
budget = 500 # Number of samples in the training and testing sets
trainingIncrement = 10 # Number of examples that are going to change in each interaction
seed = 723


digits = load_digits() # Data
clf = MultinomialNB()  # Multinomial Naive Bayes

dataset = range(0, len(digits.data))

graph = []
np.random.seed(seed)

for k in range(repetition):
 
	unlabeledData = np.random.permutation(dataset)
    
	i = trainingIncrement

	output = {'labeled' : [], 'accuracy' : []}

	labeledData = []

	while i < budget and len(unlabeledData) > 0:
		for j in range(trainingIncrement):
			labeledData += [unlabeledData.pop(j)]

		clf.fit(digits.data[data[:i]], digits.target[data[:i]])
		y = clf.predict(digits.data[data[budget - 1:]])

		output['labeled'] += [i]
		output['accuracy'] += [accuracy_score(digits.target[data[budget - 1:]], y)]

		i += trainingIncrement


	graph += [output]

output = {'labeled' : [], 'accuracy' : []}

# Calculation the average
for i in range(len(graph[0]['labeled'])):
    acc = 0
    output['labeled'] += [graph[0]['labeled'][i]]
    
    for j in range(len(graph)):
        acc += graph[j]['accuracy'][i]
    acc /= float(len(graph))
    
    output['accuracy'] += [acc]
   
graph += [output]

# Ploting
for i in range(len(graph) - 1):
    plt.plot(graph[i]['labeled'], graph[i]['accuracy'], '--', label='Test %d' % (i+1))
plt.plot(graph[i+1]['labeled'], graph[i+1]['accuracy'], '-', label='Average')
plt.legend(loc='best')
plt.show()