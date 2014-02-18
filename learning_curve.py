'''
Created on Jan 30, 2014

@author: mbilgic

For now, the program is handling just binary classification

'''

from time import time

import argparse # To use arguments
import numpy as np

from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.datasets import load_svmlight_file

from instance_strategies import LogGainStrategy, RandomStrategy, UncStrategy, RotateStrategy, BootstrapFromEach

from collections import defaultdict


if (__name__ == '__main__'):
    
    print "Loading the data"
    
    t0 = time()

    parser = argparse.ArgumentParser()

    parser.add_argument("-d", '--data', nargs=3, metavar=('pool', 'test', 'features'),
                        default=["data/imdb-binary-pool-mindf5-ng11", "data/imdb-binary-test-mindf5-ng11" ,27272],
                        help='Files that contains the data, pool and test, and number of \
                        features (default: data/imdb-binary-pool-mindf5-ng11 data/imdb-binary-test-mindf5-ng11 27272).')
    parser.add_argument("-nt", "--numTrials", type=int, default=10, help="Number of trials (default: 10).")
    parser.add_argument("-st", "--strategy", choices=['loggain', 'rand','unc'], default='rand',
                        help="Represent the base strategy for choosing next samples (default: rand).")
    parser.add_argument("-s", '--sizes', nargs=4, metavar=('bootstrap', 'budget', 'stepsize', 'subpool'),
                        default=[2, 500, 2, 250], type=int, help='Bootsrap, budget, \
                        step size and sub pool (default: 10 510 10 250).')

    args = parser.parse_args()

    dataPool = args.data[0]
    dataTest = args.data[1]
    n_features = args.data[2]

    X_pool, y_pool = load_svmlight_file(dataPool, n_features=n_features)
    X_test, y_test = load_svmlight_file(dataTest, n_features=n_features)

    duration = time() - t0
    
    num_pool, num_feat = X_pool.shape

    print
    print "Loading took %0.2fs." % duration
    print

    numtrials = args.numTrials
    strategy = args.strategy

    bootStrapSize = args.sizes[0]
    budget = args.sizes[1]
    stepSize = args.sizes[2]
    sub_pool = args.sizes[3]
    
    alpha=1
    
    accuracies = defaultdict(lambda: [])
    
    aucs = defaultdict(lambda: [])    
    
    num_test = X_test.shape[0]
    
    t0 = time()
    
    for t in range(numtrials):
        
        print "trial", t
        
        X_pool_csr = X_pool.tocsr()
    
        pool = set(range(len(y_pool)))
        
        trainIndices = []
        
        bootsrapped = False

        if strategy == 'loggain':
            activeS = LogGainStrategy(classifier=MultinomialNB, seed=t, sub_pool=sub_pool, alpha=alpha)
        elif strategy == 'rand':    
            activeS = RandomStrategy(seed=t)
        elif strategy == 'unc':
            activeS = UncStrategy(seed=t, sub_pool = sub_pool)

        
        model = None
                    
        while len(trainIndices) < budget and len(pool) > stepSize:
            
            
            if not bootsrapped:
                bootS = BootstrapFromEach(t)
                newIndices = bootS.bootstrap(pool, y=y_pool, k=bootStrapSize)
                bootsrapped = True
            else:
                newIndices = activeS.chooseNext(pool, X_pool_csr, model, k = stepSize, current_train_indices = trainIndices, current_train_y = y_pool[trainIndices])
            
            pool.difference_update(newIndices)
            
            trainIndices.extend(newIndices)
    
            model = MultinomialNB(alpha=alpha)
            
            model.fit(X_pool_csr[trainIndices], y_pool[trainIndices])
            
           
    
            y_probas = model.predict_proba(X_test)
            auc = metrics.roc_auc_score(y_test, y_probas[:,1])     
            
            pred_y = model.classes_[np.argmax(y_probas, axis=1)]
            
            accu = metrics.accuracy_score(y_test, pred_y)
            
            accuracies[len(trainIndices)].append(accu)
            aucs[len(trainIndices)].append(auc)
    
    duration = time() - t0
    
    
    # print the accuracies
    
    x = sorted(accuracies.keys())
    y = [np.mean(accuracies[xi]) for xi in x]
    z = [np.std(accuracies[xi]) for xi in x]
    
    print
    print "Train_size\tAccu_Mean\tAccu_Std"
    for a, b, c in zip(x, y, z):
        print "%d\t%0.3f\t%0.3f" % (a, b, c)
    
    x = sorted(aucs.keys())
    y = [np.mean(aucs[xi]) for xi in x]
    z = [np.std(aucs[xi]) for xi in x]
    
    print
    print "Train_size\tAUC_Mean\tAUC_Std"
    for a, b, c in zip(x, y, z):
        print "%d\t%0.3f\t%0.3f" % (a, b, c)
        
    duration = time() - t0

    print
    print "Learning curve took %0.2fs." % duration
    print
    

