# -*- coding: utf-8 -*-
import numpy as np
import sklearn.model_selection


# logestic regesssion,similar used in coding assigment 2
def sigmold(t_value):
    return 1/(1+np.exp(-t_value))

def softmax(z):
    # print(z.shape)
    e_z = np.exp(z - np.ndarray.max(z,axis=1).reshape(-1,1))

    z_softmax = np.divide(e_z ,np.sum(e_z,axis=1).reshape((-1,1))) # softmax
    return z_softmax.T

def cot(actual, predict):
    return -np.sum(actual*np.log(predict))

def cross_entropy_loss(predict, actural):
    ss = predict.shape
    
    if ((ss[1]) == 0):
        print(ss)
    loss = -1/(ss[1])*np.sum(actural.T*np.log(predict+1e-12)) 

    return loss
    

def OneHot(yloc):
    marker =  np.zeros_like(yloc)
    maxloc = np.argmax(yloc,axis=1)

    for i in range(maxloc.shape[0]):
         marker[i,maxloc[i]] = 1
    return maxloc,marker
    

    
def predict(X_loc, W, t_loc=None):
    # X_new: Nsample x (d+1)
    # W: (d+1) x K

    # TODO Your code here
    z = np.dot(X_loc,W)

    # # get the softmax
    y_s = softmax(z)

    # one_hot
    maxlocation,marker = OneHot(y_s.T)
    
    t_marker =  np.zeros_like(marker)

    t_loc = t_loc.T.astype(int)
    #t_marker[:,t_loc] = 1
    for i in range(t_loc.shape[0]):
         t_marker[i,t_loc[i]] = 1

    gradient =  (marker -t_marker)
    loss = cross_entropy_loss(y_s,t_marker)

    # get currect answer %
    acc = np.sum((maxlocation.reshape(-1,1)==t_loc.reshape(-1,1)).astype(int))/X_loc.shape[0]

    return marker, t_marker, gradient, acc,loss


def train(X_train, t_train, X_val, t_val,n_poch):

    N_train = X_train.shape[0]
    N_val = X_val.shape[0]

    #####################
    # setting
    N_class = 2

    alpha = 0.1      # learning rate
    batch_size = 200    # batch size
    MaxEpoch = n_poch       # Maximum epoch
    decay = 0.          # weight decay

    # initialization
    w = np.zeros([X_train.shape[1], N_class]) # 10 feacture
    # w: (d+1)x1
    # print(w.shape)

    train_losses = []
    valid_accs = []

    w_best = None
    epoch_best = 0
    acc_best = 0

    for epoch in range(MaxEpoch):
        loss_this_epoch = 0
        for b in range(int(np.ceil(N_train/batch_size))):

            X_batch = X_train[b*batch_size: (b+1)*batch_size]
            t_batch = t_train[b*batch_size: (b+1)*batch_size]

            y_predict,t_hat, gradient, acc,loss = predict(X_batch, w, t_batch)
            loss_this_epoch += loss

            # TODO: Your code here
            # Mini-batch gradient descent
            diff = np.dot(X_batch.T,gradient)
            w=w-(alpha/X_batch.shape[0])*diff
            
        # TODO: Your code here
        # monitor model behavior after each epoch
        # 1. Compute the training loss by averaging loss_this_epoch
        avg_training_loss = np.sum(np.absolute(loss_this_epoch))/int(batch_size)
        train_losses.append(avg_training_loss)
        
        
        # 2. Perform validation on the validation set by the risk
        y_predict,t_hat, gradient, acc,loss = predict(X_val, w, t_val)
        valid_accs.append(acc)
        
        
        # 3. Keep track of the best validation epoch, risk, and the weights
        if (len(train_losses)<=1):
            acc_best = acc
            w_best=w
            epoch_best=epoch
        elif (acc_best< acc):
            acc_best = acc
            w_best=w
            epoch_best=epoch



    return epoch_best, acc_best,  w_best, train_losses, valid_accs


def runRegreesion(times):
    X_train = np.loadtxt("X_train.txt")
    t_train = np.loadtxt("t_train.txt")
    X_test = np.loadtxt("X_test.txt")
    t_test = np.loadtxt("t_test.txt")

    X_train, X_val, t_train, t_val = sklearn.model_selection.train_test_split(X_train, t_train, test_size = 0.1, random_state = 0)

    spt = int(X_train.shape[0]*0.9)

    X_val = X_train[spt:]
    t_val = t_train[spt:]

    X_train = X_train[:spt]
    t_train = t_train[:spt]

    # TODO: report 3 number, plot 2 curves
    epoch_best, acc_best,  W_best, train_losses, valid_accs = train(X_train, t_train, X_val, t_val,times)


    _, _, _, acc_test,_ = predict(X_test, W_best, t_test)

    w_importance = np.absolute(W_best[:,0])

    return acc_best,valid_accs,w_importance


if __name__ == "__main__":
    bestScore,prediction_accurcy,weightImportance = runRegreesion(10)
    print(bestScore,weightImportance)
