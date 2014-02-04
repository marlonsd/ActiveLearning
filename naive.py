#!/usr/bin/python

import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.datasets import *
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import KFold
import matplotlib.pyplot as plt # Ploting

fold = 5 # Number of folds in the cross-validation
budget = 898 # Number of samples in the training and testing sets
trainingIncrement = 10 # Number of examples that are going to change in each interaction
seed = 723

def uncertaintySampling(unlabeled, labeled, dataset, trainingIncrement):
	candidates = []
	for i in range(len(unlabeled)):
		count = 1 - (len(np.where(labeled == unlabeled[i])) / float(len(labeled))) 
		candidates += [[i, count]]

	candidates = sorted(candidates, key= lambda (x,y):y, reverse=False)

	out = []

	for i in range(trainingIncrement):
		out += [candidates[i][0]]

	return out

digits = load_digits() # Data
clf = MultinomialNB()  # Multinomial Naive Bayes

dataset = range(0, len(digits.data))

graph = []
np.random.seed(seed)

kf = KFold(len(digits.target), n_folds=fold, indices=True)

for unlabeledData, test in kf:
	
	unlabeledData = np.random.permutation(unlabeledData)

	i = trainingIncrement

	output = {'labeled' : [], 'accuracy' : []}

	labeledData = unlabeledData[:trainingIncrement]

	while i < budget and len(unlabeledData) > 0:

		if len(labeledData) == trainingIncrement:
			for j in range (trainingIncrement):
				unlabeledData = np.delete(unlabeledData, 0)

		clf.fit(digits.data[labeledData], digits.target[labeledData])
		y = clf.predict(digits.data[test])

		output['labeled'] += [i]
		output['accuracy'] += [accuracy_score(digits.target[test], y)]

		# print 'Labeled Data: ', labeledData
		# print
		# print 'Unlabeled Data: ', unlabeledData

		# print
		# print

		i += trainingIncrement
		append = uncertaintySampling(unlabeledData, labeledData, digits, trainingIncrement)
		labeledData = np.append(labeledData, unlabeledData[append])
		unlabeledData = np.delete(unlabeledData, append)


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