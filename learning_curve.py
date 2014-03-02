'''
Created on Jan 30, 2014

@author: mbilgic

For now, the program is handling just binary classification

'''

from time import time

import argparse # To use arguments
import math
import numpy as np
import sys

from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.datasets import load_svmlight_file

from instance_strategies import LogGainStrategy, RandomStrategy, UncStrategy, RotateStrategy, BootstrapFromEach, QBCStrategy

from collections import defaultdict

import matplotlib.pyplot as plt # Plotting


def learning(num_trials, X_pool, y_pool, strategy, budget, step_size, boot_strap_size):
    accuracies = defaultdict(lambda: [])
    aucs = defaultdict(lambda: [])    

    for t in range(num_trials):
        
        print "trial", t
        
        X_pool_csr = X_pool.tocsr()
    
        pool = set(range(len(y_pool)))
        
        trainIndices = []
        
        bootsrapped = False

        # Choosing strategy
        if strategy == 'loggain':
            active_s = LogGainStrategy(classifier=MultinomialNB, seed=t, sub_pool=sub_pool, alpha=alpha)
        elif strategy == 'qbc':
            active_s = QBCStrategy(classifier=MultinomialNB, classifier_args=alpha)
        elif strategy == 'rand':    
            active_s = RandomStrategy(seed=t)
        elif strategy == 'unc':
            active_s = UncStrategy(seed=t, sub_pool = sub_pool)

        
        model = None

        # Loop for prediction
        while len(trainIndices) < budget and len(pool) > step_size:
            
            if not bootsrapped:
                boot_s = BootstrapFromEach(t)
                newIndices = boot_s.bootstrap(pool, y=y_pool, k=boot_strap_size)
                bootsrapped = True
            else:
                newIndices = active_s.chooseNext(pool, X_pool_csr, model, k = step_size, current_train_indices = trainIndices, current_train_y = y_pool[trainIndices])
            
            pool.difference_update(newIndices)
            
            trainIndices.extend(newIndices)
    
            model = MultinomialNB(alpha=alpha)
            
            model.fit(X_pool_csr[trainIndices], y_pool[trainIndices])
            
           # Prediction
            y_probas = model.predict_proba(X_test)

            # Metrics
            auc = metrics.roc_auc_score(y_test, y_probas[:,1])     
            
            pred_y = model.classes_[np.argmax(y_probas, axis=1)]
            
            accu = metrics.accuracy_score(y_test, pred_y)
            
            accuracies[len(trainIndices)].append(accu)
            aucs[len(trainIndices)].append(auc)

    return accuracies, aucs
    

if (__name__ == '__main__'):
    
    print "Loading the data"
    
    t0 = time()

    ### Arguments Treatment ###
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument("-d", '--data', nargs=2, metavar=('pool', 'test'),
                        default=["data/imdb-binary-pool-mindf5-ng11", "data/imdb-binary-test-mindf5-ng11"],
                        help='Files that contains the data, pool and test, and number of \
                        features (default: data/imdb-binary-pool-mindf5-ng11 data/imdb-binary-test-mindf5-ng11 27272).')

    # Number of Trials
    parser.add_argument("-nt", "--num_trials", type=int, default=10, help="Number of trials (default: 10).")

    # Strategies
    # Usage: -st rand qbc
    parser.add_argument("-st", "--strategies", choices=['loggain', 'qbc', 'rand','unc'], nargs='*',default='rand',
                        help="Represent a list of strategies for choosing next samples (default: rand).")

    # Boot Strap
    parser.add_argument("-bs", '--bootstrap', default=10, type=int, 
                        help='Sets the Boot strap (default: 10).')
    
    # Budget
    parser.add_argument("-b", '--budget', default=500, type=int,
                        help='Sets the budget (default: 500).')

    # Step size
    parser.add_argument("-sz", '--stepsize', default=10, type=int,
                        help='Sets the step size (default: 10).')

    # Sub pool size
    parser.add_argument("-sp", '--subpool', default=250, type=int,
                        help='Sets the sub pool size (default: 250).')

    args = parser.parse_args()

    data_pool = args.data[0]
    data_test = args.data[1]

    X_pool, y_pool = load_svmlight_file(data_pool)
    num_pool, num_feat = X_pool.shape

    X_test, y_test = load_svmlight_file(data_test, n_features=num_feat)

    duration = time() - t0

    print
    print "Loading took %0.2fs." % duration
    print

    num_trials = args.num_trials
    strategies = args.strategies

    boot_strap_size = args.bootstrap
    budget = args.budget
    step_size = args.stepsize
    sub_pool = args.subpool
    
    alpha=1
    
    duration = defaultdict(lambda: 0.0)

    accuracies = defaultdict(lambda: [])
    
    aucs = defaultdict(lambda: [])    
    
    num_test = X_test.shape[0]

    # Main Loop
    for strategy in strategies:
        t0 = time()

        accuracies[strategy], aucs[strategy] = learning(num_trials, X_pool, y_pool, strategy, budget, step_size, boot_strap_size)

        duration[strategy] = time() - t0

        print
        print "%s Learning curve took %0.2fs." % (strategy, duration[strategy])
        print
    
    
    values = sorted(accuracies[strategies[0]].keys())

    # print the accuracies
    print
    print "\t\tAccuracy mean"
    print "Train Size\t",
    for strategy in strategies:
        print "%s\t\t" % strategy,
    print

    for value in values:
        print "%d\t\t" % value,
        for strategy in strategies:
            print "%0.3f\t\t" % np.mean(accuracies[strategy][value]),
        print
        
    # print the aucs
    print
    print "\t\tAUC mean"
    print "Train Size\t",
    for strategy in strategies:
        print "%s\t\t" % strategy,
    print

    for value in values:
        print "%d\t\t" % value,
        for strategy in strategies:
            print "%0.3f\t\t" % np.mean(aucs[strategy][value]),
        print

    # print the times
    print
    print "\t\tTime"
    print "Strategy\tTime"

    for strategy in strategies:
        print "%s\t%0.2f" % (strategy, duration[strategy])

    # plotting
    for strategy in strategies:
        accuracy = accuracies[strategy]
        auc = aucs[strategy]


        x = sorted(accuracy.keys())
        y = [np.mean(accuracy[xi]) for xi in x]
        z = [np.std(accuracy[xi]) for xi in x]
        e = np.array(z) / math.sqrt(num_trials)
        
        # print
        # print "Train_size\tAccu_Mean\tAccu_Std"
        # for a, b, c in zip(x, y, z):
        #     print "%d\t%0.3f\t%0.3f" % (a, b, c)

        plt.figure(1)
        plt.subplot(211)
        # plt.errorbar(x,y,yerr=e, label=strategy)
        plt.plot(x, y, '-', label=strategy)
        plt.legend(loc='best')
        plt.title('Accuracy')

        x = sorted(auc.keys())
        y = [np.mean(auc[xi]) for xi in x]
        z = [np.std(auc[xi]) for xi in x]
        e = np.array(z) / math.sqrt(num_trials)
        
        # print
        # print "Train_size\tAUC_Mean\tAUC_Std"
        # for a, b, c in zip(x, y, z):
        #     print "%d\t%0.3f\t%0.3f" % (a, b, c)
            

        plt.subplot(212)
        # plt.errorbar(x,y,yerr=e, label=strategy)
        plt.plot(x, y, '-', label=strategy)
        plt.legend(loc='best')
        plt.title('AUC')

    plt.show()
    

