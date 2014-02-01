#!/usr/bin/python

import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.datasets import *
import matplotlib.pyplot as plt # Ploting

repetition = 5 # The algorithm will run this number of times
dataSplit = 500 # Number of samples in the training and testing sets
trainingIncrement = 10 # Number of examples that are going to change in each interaction

# Split the data randomly
def randonization(data):
	randomData = np.random.permutation(data)

	train = {'sample' : [], 'answer' : []}
	test = {'sample' : [], 'answer' : []}
    
	for i in range (dataSplit):
		train['sample'] += [randomData[i][0]]
		test['sample'] += [randomData[i+dataSplit][0]]
        
		train['answer'] += [randomData[i][1]]
		test['answer'] += [randomData[i+dataSplit][1]]

	return [train, test]

digits = load_digits() # Data
clf = MultinomialNB()  # Multinomial Naive Bayes

data = []

# Transforming the dataset in a list [[SAMPLE, CLASSIFICATION]]
for i in range (len(digits.data)):
    data += [[digits.data[i], digits.target[i]]]

graph = []

for k in range(repetition):
 
	train, test = randonization(data)
    
	output = {'labeled' : [], 'accuracy' : []}
	i = trainingIncrement

	while i < dataSplit:
		accuracy = 0
		# Training
		clf.fit(train['sample'][:i], train['answer'][:i])

		# Predicting
		for j in range (dataSplit):
			if clf.predict(test['sample'][j]) == test['answer'][j]:
				accuracy += 1.0

		output['labeled'] += [i]
		output['accuracy'] += [accuracy]

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
    plt.plot(graph[i]['labeled'], graph[i]['accuracy'], '--')
plt.plot(graph[i+1]['labeled'], graph[i+1]['accuracy'], '-')
plt.show()