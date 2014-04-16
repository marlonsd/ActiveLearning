from collections import defaultdict
from sklearn import preprocessing
import numpy as np
import arff

def conversion_features(dataset):
	features = defaultdict(lambda: [])

	to_categorize = []
	to_binarize = []

	categ_features = []
	binary_features = []

	lb = preprocessing.LabelBinarizer()
	min_max_scaler = preprocessing.MinMaxScaler()
	binarizer = preprocessing.Binarizer()

	# Converting features into string or float
	for data in dataset:
		for pos, feature in enumerate(data):
			try:
				feature = float(feature)
				if (feature != 0 or feature != 1):
					# to_binarize.append(pos)
					pass
			except:
				feature = str(feature)
				to_categorize.append(pos)
			features[pos].append(feature)

	# print features

	to_categorize = set(to_categorize)
	to_binarize = set(to_binarize)

	# Dealing with categorical features
	for pos, values in features.items():

		if pos in to_categorize:
			new_values = lb.fit(values).transform(values)
			print lb.classes_

			del features[pos]

			new_values = new_values.tolist()
			print new_values
			categ_features.append(new_values)
			print
		elif pos in to_binarize:
			# print '- Binary'
			new_values = preprocessing.StandardScaler().fit_transform(np.array([values]))
			# print new_values
			new_values = min_max_scaler.fit_transform(new_values)
			# print new_values
			new_values = binarizer.transform(new_values)

			new_array = [i for i in new_values[0]]

			features[pos] = new_array

			# print new_array

	print
	print features

	# Organizing array of features

	features_name = features.keys()
	name = features_name.pop(0)

	new_features = map(list,zip(features[name]))
	print new_features

	del features[name]

	if (len(features) > 0):
		for i in range(len(new_features)):
			for name in features_name:
				new_features[i] += [features[name].pop(0)]

	print 
	# print new_features

	print categ_features
	print 

	for feature in categ_features:
		for i, pos in enumerate(feature):
			for value in pos:
				new_features[i].append(value)

	print new_features

	return np.array(new_features)


def load_arff(filename):

	array = []
	for row in (arff.load(filename)):
		array.append(list(row))

	nparray = np.array(array)

	x = []
	y = []

	for element in nparray:
		x.append(element[:-1])
		y.append(element[-1])

	x = np.array(x)
	y = np.array(y)

	return x, y

def main():
	x, y = load_arff('data/example.arff')
	print x
	print
	x = conversion_features(x)

if __name__=="__main__":
    main()