import numpy as np
import math

from sklearn.naive_bayes import MultinomialNB
# from sklearn.datasets import load_svmlight_file
from sklearn.datasets import *

# from instance_strategies import LogGainStrategy, RandomStrategy, UncStrategy, RotateStrategy, BootstrapFromEach

from collections import defaultdict

seed = 723
np.random.seed(seed)
digits = load_digits() # Data

def entropy(sample):
	index = defaultdict(lambda: 0)
	size = float(len(sample))

	for i in sample:
		index[i] += 1

	# print index
	out = 0
	for i in index:
		aux = (float(index[i]/size))
		# print index[i], aux
		out += (aux*math.log(aux, 2))

	return -out


def QueryByCommittee (pool, model, X=None, k=1, current_train_indices=None, current_train_y=None, c=5):
	l = []
	dataset = range(0, len(pool))

	for i in range(c):
		aux = np.random.randint(0, len(pool), size=len(pool))
		
		l_aux = []
		for j in aux:
			l_aux += [pool[j]]
		l += [l_aux]

	for item in l:
		print item

	print 

	i = 0
	out_aux = []
	for sample in l:
		y = model.predict(digits.data[sample])
		print y
		out_aux += [[sample, y, entropy(y)]]
		i += 1

	out_aux = sorted(out_aux, key = lambda (x): x[2], reverse = True)

	out = []

	for i in range(c):
		out += [out_aux[i][0]]

	return out

clf = MultinomialNB()  # Multinomial Naive Bayes
labeledData = np.arange(30, 40)

clf.fit(digits.data[30:40], digits.target[30:40])

y = clf.predict(digits.data[30:40])
print y

print clf
print

QueryByCommittee(labeledData, clf, c=5)