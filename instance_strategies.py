'''
Created on Jan 28, 2014

@author: mbilgic

For now, the program is handling just binary classification
'''

import math
import numpy as np
import scipy.sparse as ss
from collections import defaultdict

class RandomBootstrap(object):
    def __init__(self, seed):
        self.randS = RandomStrategy(seed)
        
    def bootstrap(self, pool, y=None, k=1):
        return self.randS.chooseNext(pool, k=k)

class BootstrapFromEach(object):
    def __init__(self, seed):
        self.randS = RandomStrategy(seed)
        
    def bootstrap(self, pool, y, k=1):
        data = defaultdict(lambda: [])
        for i in pool:
            data[y[i]].append(i)
        chosen = []
        num_classes = len(data.keys())
        for label in data.keys():
            candidates = data[label]
            indices = self.randS.chooseNext(candidates, k=k/num_classes)
            chosen.extend(indices)
        return chosen


class BaseStrategy(object):
    
    def __init__(self, seed=0):
        self.randgen = np.random
        self.randgen.seed(seed)
        
    def chooseNext(self, pool, X=None, model=None, k=1, current_train_indices = None, current_train_y = None):
        pass

class RandomStrategy(BaseStrategy):
        
    def chooseNext(self, pool, X=None, model=None, k=1, current_train_indices = None, current_train_y = None):
        list_pool = list(pool)
        rand_indices = self.randgen.permutation(len(pool))
        return [list_pool[i] for i in rand_indices[:k]]

class UncStrategy(BaseStrategy):
    
    def __init__(self, seed=0, sub_pool = None):
        super(UncStrategy, self).__init__(seed=seed)
        self.sub_pool = sub_pool
    
    def chooseNext(self, pool, X=None, model=None, k=1, current_train_indices = None, current_train_y = None):
        
        num_candidates = len(pool)
        
        if self.sub_pool is not None:
            num_candidates = self.sub_pool
        
        rand_indices = self.randgen.permutation(len(pool))        
        list_pool = list(pool)        
        candidates = [list_pool[i] for i in rand_indices[:num_candidates]]
        
        if ss.issparse(X):
            if not ss.isspmatrix_csr(X):
                X = X.tocsr()
        
        probs = model.predict_proba(X[candidates])        
        uncerts = np.min(probs, axis=1)        
        uis = np.argsort(uncerts)[::-1]
        chosen = [candidates[i] for i in uis[:k]]       
        return chosen

class QBCStrategy(BaseStrategy):
    
    def __init__(self, classifier, classifier_args, seed=0, sub_pool = None, num_committee = 4):
        super(QBCStrategy, self).__init__(seed=seed)
        self.sub_pool = sub_pool
        self.num_committee = num_committee
        self.classifier = classifier
        self.classifier_args = classifier_args
        
    
    def chooseNext(self, pool, X=None, model=None, k=1, current_train_indices = None, current_train_y = None):
         
        def entropy(sample):
            index = defaultdict(lambda: 0.0)
            size = float(len(sample))

            for i in sample:
                index[i] += 1.0

            out = 0
            for i in index:
                aux = (float(index[i]/size))
                out += (aux*math.log(aux, 2))

            return -out

        num_candidates = len(pool)
        
        if self.sub_pool is not None:
            num_candidates = self.sub_pool
        
        rand_indices = self.randgen.permutation(len(pool))        
        list_pool = list(pool)        
        candidates = [list_pool[i] for i in rand_indices[:num_candidates]]
        
        if ss.issparse(X):
            if not ss.isspmatrix_csr(X):
                X = X.tocsr()
        
        # Create bags
        
        comm_predictions = []
        
        for c in range(self.num_committee):
            r_inds = self.randgen.randint(0, len(current_train_indices), size=len(current_train_indices))
            bag = [current_train_indices[i] for i in r_inds]
            bag_y = [current_train_y[i] for i in r_inds]
            new_classifier = self.classifier(self.classifier_args)
            new_classifier.fit(X[bag], bag_y)
            
            predictions = new_classifier.predict(X[candidates])
            
            comm_predictions.append(predictions)
        
        # Compute disagreement for com_predictions

        candidates_predictions = []
        for i in range(len(comm_predictions[0])):
            aux_candidates = []
            for prediction in comm_predictions:
                aux_candidates.append(prediction[i])
            disagreement = entropy(aux_candidates)
            candidates_predictions.append([i, disagreement])

        # choose the ones that are most disagreed
        candidates_predictions = sorted(candidates_predictions, key=lambda x:x[1], reverse=True)
        chosen = [candidates[position[0]] for position in candidates_predictions[:k]]

        
        return chosen

class LogGainStrategy(BaseStrategy):
    
    def __init__(self, classifier, seed = 0, sub_pool = None, **classifier_args):
        super(LogGainStrategy, self).__init__(seed=seed)
        self.classifier = classifier
        self.sub_pool = sub_pool
        self.classifier_args = classifier_args
    
    def log_gain(self, probs, labels):
        lg = 0
        for i in xrange(len(probs)):
            lg -= np.log(probs[i][labels[i]])
        return lg
    
    def chooseNext(self, pool, X=None, model=None, k=1, current_train_indices = None, current_train_y = None):
        
        num_candidates = len(pool)
        
        if self.sub_pool is not None:
            num_candidates = self.sub_pool
        
        list_pool = list(pool)
        
        
        #random candidates
        rand_indices = self.randgen.permutation(len(pool))                
        candidates = [list_pool[i] for i in rand_indices[:num_candidates]]
        
        if ss.issparse(X):
            if not ss.isspmatrix_csr(X):
                X = X.tocsr()
                        
        cand_probs = model.predict_proba(X[candidates])    
        
        utils = []
        
        for i in xrange(num_candidates):
            #assume binary
            new_train_inds = list(current_train_indices)
            new_train_inds.append(candidates[i])
            util = 0
            for c in [0, 1]:
                new_train_y = list(current_train_y)
                new_train_y.append(c)
                new_classifier = self.classifier(**self.classifier_args)
                new_classifier.fit(X[new_train_inds], new_train_y)
                new_probs = new_classifier.predict_proba(X[current_train_indices])
                util += cand_probs[i][c] * self.log_gain(new_probs, current_train_y)
            
            utils.append(util)
        
        uis = np.argsort(utils)
        
        
        chosen = [candidates[i] for i in uis[:k]]
        
        return chosen

class RotateStrategy(BaseStrategy):
    
    def __init__(self, strategies):
        super(RotateStrategy, self).__init__(seed=0)
        self.strategies = strategies
        self.counter = -1
    
    def chooseNext(self, pool, X=None, model=None, k=1, current_train_indices = None, current_train_y = None):
        self.counter = (self.counter+1) % len(self.strategies)
        return self.strategies[self.counter].chooseNext(pool, X, model, k=k, current_train_indices = current_train_indices, current_train_y = current_train_y)
                

        