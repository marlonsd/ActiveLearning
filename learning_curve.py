'''
Created on Jan 30, 2014

@author: mbilgic
'''

from time import time
from sys import platform # To check operating systems

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
           

    # Checking operating system. Accessing folders is different in Windows and Unix based systems
    if platform == "win32":
        X_pool, y_pool = load_svmlight_file("data\\imdb-binary-pool-mindf5-ng11", n_features=27272)
        X_test, y_test = load_svmlight_file("data\\imdb-binary-test-mindf5-ng11", n_features=27272)
        
    elif platform == "linux" or platform == "linux2" or platform == "darwin":
        X_pool, y_pool = load_svmlight_file("data/imdb-binary-pool-mindf5-ng11", n_features=27272)
        X_test, y_test = load_svmlight_file("data/imdb-binary-test-mindf5-ng11", n_features=27272)
    
    duration = time() - t0
    
    num_pool, num_feat = X_pool.shape

    print
    print "Loading took %0.2fs." % duration
    print
    
    bootStrapSize = 2
    budget = 500
    stepSize = 2
    numtrials = 10
    sub_pool = 250
    
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
        
        activeS = RandomStrategy(seed=t)
        #activeS = UncStrategy(seed=t, sub_pool = sub_pool)       
        #activeS = LogGainStrategy(classifier=MultinomialNB, seed=t, sub_pool=sub_pool, alpha=alpha)
        #activeS = RotateStrategy(strategies = [UncStrategy(seed=t, sub_pool = sub_pool), LogGainStrategy(classifier=MultinomialNB, seed=t, sub_pool=sub_pool, alpha=alpha)])

        
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
            
    
            #print "train size:\t%d" % len(trainIndices)
            #print "accu:\t%0.3f" % accu
            #print "auc:\t%0.3f" % auc
            #print
            
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
        #print "%0.3f" % b
        print "%d\t%0.3f\t%0.3f" % (a, b, c)
    
    x = sorted(aucs.keys())
    y = [np.mean(aucs[xi]) for xi in x]
    z = [np.std(aucs[xi]) for xi in x]
    
    print
    print "Train_size\tAUC_Mean\tAUC_Std"
    for a, b, c in zip(x, y, z):
        #print "%0.3f" % b
        print "%d\t%0.3f\t%0.3f" % (a, b, c)
        
    duration = time() - t0

    print
    print "Learning curve took %0.2fs." % duration
    print
    

