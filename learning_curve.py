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

import matplotlib.pyplot as plt # Plotting


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

    # Strategy
    parser.add_argument("-st", "--strategy", choices=['loggain', 'rand','unc'], default='rand',
                        help="Represent the base strategy for choosing next samples (default: rand).")

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
    strategy = args.strategy

    boot_strap_size = args.bootstrap
    budget = args.budget
    step_size = args.stepsize
    sub_pool = args.subpool
    
    alpha=1
    
    accuracies = defaultdict(lambda: [])
    
    aucs = defaultdict(lambda: [])    
    
    num_test = X_test.shape[0]
    
    t0 = time()
    
    # Main Loop
    for t in range(num_trials):
        
        print "trial", t
        
        X_pool_csr = X_pool.tocsr()
    
        pool = set(range(len(y_pool)))
        
        trainIndices = []
        
        bootsrapped = False

        # Choosing strategy
        if strategy == 'loggain':
            activeS = LogGainStrategy(classifier=MultinomialNB, seed=t, sub_pool=sub_pool, alpha=alpha)
        elif strategy == 'rand':    
            activeS = RandomStrategy(seed=t)
        elif strategy == 'unc':
            activeS = UncStrategy(seed=t, sub_pool = sub_pool)

        
        model = None
        
        # Loop for prediction
        while len(trainIndices) < budget and len(pool) > step_size:
            
            if not bootsrapped:
                bootS = BootstrapFromEach(t)
                newIndices = bootS.bootstrap(pool, y=y_pool, k=boot_strap_size)
                bootsrapped = True
            else:
                newIndices = activeS.chooseNext(pool, X_pool_csr, model, k = step_size, current_train_indices = trainIndices, current_train_y = y_pool[trainIndices])
            
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
    
    duration = time() - t0
    
    
    # print the accuracies
    
    x = sorted(accuracies.keys())
    y = [np.mean(accuracies[xi]) for xi in x]
    z = [np.std(accuracies[xi]) for xi in x]
    
    print
    print "Train_size\tAccu_Mean\tAccu_Std"
    for a, b, c in zip(x, y, z):
        print "%d\t%0.3f\t%0.3f" % (a, b, c)

    x2 = sorted(aucs.keys())
    y2 = [np.mean(aucs[xi]) for xi in x]
    z2 = [np.std(aucs[xi]) for xi in x]
    
    print
    print "Train_size\tAUC_Mean\tAUC_Std"
    for a, b, c in zip(x2, y2, z2):
        print "%d\t%0.3f\t%0.3f" % (a, b, c)
        
    duration = time() - t0

    plt.figure(1)
    plt.subplot(211)
    plt.plot(x, y, 'bo', x, z, 'k')
    plt.title('Fig 1')

    plt.figure(2)
    plt.subplot(212)
    plt.plot(x2, y2, 'r--', x2, z2, 'k')
    plt.title('Fig 2')

    plt.show()

    print
    print "Learning curve took %0.2fs." % duration
    print
    

