# -*- coding: utf-8 -*-
import numpy as np
# scikit learn split data
import sklearn.model_selection

# lienear regesssuib,similar used in coding assigment 2
def predict(X_loc, W, t_loc=None):

    # TODO Your code here
    y = np.dot(X_loc,W)
    pret_t = 1*(y >0.5)
    acc = sklearn.metrics.accuracy_score(pret_t.T.ravel(), t_loc.T.ravel())
    diff = pret_t-t_loc
    return  acc,diff


def train(X_train, t_train, X_val, t_val,n_poch):

    N_train = X_train.shape[0]
    N_val = X_val.shape[0]

    #####################
    # setting
    N_class = 1

    alpha = 0.01      # learning rate
    batch_size = 10000    # batch size
    MaxEpoch = n_poch       # Maximum epoch
    decay = 0.          # weight decay

    # initialization
    w = np.zeros([X_train.shape[1], N_class]) # 10 feacture

    valid_accs = []

    w_best = None
    epoch_best = 0
    acc_best = 0

    for epoch in range(MaxEpoch):
        loss_this_epoch = 0
        for b in range(int(np.ceil(N_train/batch_size))):

            X_batch = X_train[b*batch_size: (b+1)*batch_size]
            t_batch = t_train[b*batch_size: (b+1)*batch_size].reshape(-1,1)

            acc,diff = predict(X_batch, w, t_batch)

            # TODO: Your code here
            # Mini-batch gradient descent
            diff1 = np.dot(X_batch.T,diff)
            w=w-(alpha/X_batch.shape[0])*diff1
            
        # TODO: Your code here
        # monitor model behavior after each epoch
        # 2. Perform validation on the validation set by the risk
        acc,diff = predict(X_val, w, t_val.reshape(-1,1))
        valid_accs.append(acc)
        
        # 3. Keep track of the best validation epoch, risk, and the weights
        if (len(valid_accs)<=1):
            acc_best = acc
            w_best=w
            epoch_best=epoch
        elif (acc_best< acc):
            acc_best = acc
            w_best=w
            epoch_best=epoch


    return epoch_best, acc_best,  w_best, valid_accs


def runRegreesion(times):
    X_train = np.loadtxt("X_train.txt")
    t_train = np.loadtxt("t_train.txt")
    X_test = np.loadtxt("X_test.txt")
    t_test = np.loadtxt("t_test.txt")

    # TODO: report 3 number, plot 2 curves
    epoch_best, acc_best,  W_best, valid_accs = train(X_train, t_train, X_test, t_test,times)

    valid_accs,diff = predict(X_test, W_best, t_test.reshape(-1,1))

    w_importance = np.absolute(W_best[:,0])

    return acc_best,valid_accs,w_importance


if __name__ == "__main__":
    bestScore,test_accurcy,weightImportance = runRegreesion(200)
    print(bestScore,weightImportance)
    print(np.max(test_accurcy))