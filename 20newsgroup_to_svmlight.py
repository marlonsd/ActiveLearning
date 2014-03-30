from sklearn.datasets import fetch_20newsgroups
from sklearn.datasets import dump_svmlight_file
from sklearn.feature_extraction.text import CountVectorizer


vectorizer = CountVectorizer(min_df=1, binary=False)
categories = ['alt.atheism', 'talk.religion.misc']
remove = ('headers', 'footers', 'quotes')
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories, remove=remove)
newsgroups_test = fetch_20newsgroups(subset='test', categories=categories, remove=remove)

vectors_train = vectorizer.fit_transform(newsgroups_train.data)
vectors_test = vectorizer.fit_transform(newsgroups_test.data)

dump_svmlight_file(vectors_train.toarray(), newsgroups_train.target, 'data/20_newsgroups_train')
dump_svmlight_file(vectors_test.toarray(), newsgroups_test.target, 'data/20_newsgroups_test')