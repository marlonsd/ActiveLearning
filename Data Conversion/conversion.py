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
	min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
	binarizer = preprocessing.Binarizer()

	# Converting features into string or float
	for data in dataset:
		for pos, feature in enumerate(data):
			try:
				feature = float(feature)
				if (feature != 0 or feature != 1):
					to_binarize.append(pos)
					pass
			except:
				feature = str(feature)
				to_categorize.append(pos)
			features[pos].append(feature)

	to_categorize = set(to_categorize)
	to_binarize = set(to_binarize)

	
	for pos, values in features.items():
		# Dealing with categorical features
		if pos in to_categorize:
			new_values = lb.fit(values).transform(values)

			del features[pos]

			new_values = new_values.tolist()
			categ_features.append(new_values)

		# Binarization
		elif pos in to_binarize:
			new_values = min_max_scaler.fit([values]).transform([values])
			new_values = binarizer.transform(new_values)

			new_array = [i for i in new_values[0]]

			features[pos] = new_array

	# Organizing array of features

	features_name = features.keys()
	name = features_name.pop(0)

	new_features = map(list,zip(features[name]))

	del features[name]

	if (len(features) > 0):
		for i in range(len(new_features)):
			for name in features_name:
				new_features[i] += [features[name].pop(0)]

	for feature in categ_features:
		for i, pos in enumerate(feature):
			for value in pos:
				new_features[i].append(value)

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
	x, y = load_arff('data/credit-g.arff')
	x = conversion_features(x)
	
if __name__=="__main__":
    main()