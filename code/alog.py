import numpy as np

# scikit-machine learning library
import sklearn
import sklearn.inspection
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
import sklearn.naive_bayes
import sklearn.ensemble
import sklearn.linear_model 
import sklearn.discriminant_analysis

import logistic_regression
import linear_regression

# this run all the alogritm other than linear/logestic regression

def alogWQDA(times):
    # Weighted Quadratic Discriminant Analysis

    X_train = np.loadtxt("X_train.txt")
    t_train = np.loadtxt("t_train.txt")
    X_test = np.loadtxt("X_test.txt")
    t_test = np.loadtxt("t_test.txt")

    #training
    WQDA = sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis()
    WQDA.fit(X_train, t_train)
    
    # valdiation
    t_pred = WQDA.predict(X_test)
    score = sklearn.metrics.accuracy_score(t_test, t_pred)

    coef = sklearn.inspection.permutation_importance(WQDA, X_test, t_pred)
    return score,coef.get('importances_mean')



def alogDecisionTree(times):
    X_train = np.loadtxt("X_train.txt")
    t_train = np.loadtxt("t_train.txt")
    X_test = np.loadtxt("X_test.txt")
    t_test = np.loadtxt("t_test.txt")
    X_train, X_val, t_train, t_val = sklearn.model_selection.train_test_split(X_train, t_train, test_size = 0.1, random_state = 0)
    spt = int(X_train.shape[0]*0.9)

    X_val = X_train[spt:]
    t_val = t_train[spt:]
    t_val = t_val.T.astype(int)

    X_train = X_train[:spt]
    t_train = t_train[:spt]
    t_train = t_train.T.astype(int)

    best_score = 0
    best_dtc = None
    prediction_accurcy = []
    
    for i in range(1,times):
        #training
        dtc = sklearn.tree.DecisionTreeClassifier(max_depth=i)
        dtc.fit(X_train, t_train)
        
        # valdiation
        t_pred = dtc.predict(X_val)
        score = sklearn.metrics.accuracy_score(t_val, t_pred)
        prediction_accurcy.append(score)        
        if (best_score<score):
            best_score =score
            best_dtc = dtc

    # test
    t_test_predict = best_dtc.predict(X_test)
    score = sklearn.metrics.accuracy_score(t_test, t_test_predict)
    
    return best_score,prediction_accurcy,best_dtc.feature_importances_




def alogBayesClassifier(times):

    X_train = np.loadtxt("X_train.txt")
    t_train = np.loadtxt("t_train.txt")
    X_test = np.loadtxt("X_test.txt")
    t_test = np.loadtxt("t_test.txt")

    X_train, X_val, t_train, t_val = sklearn.model_selection.train_test_split(X_train, t_train, test_size = 0.1, random_state = 0)

    spt = int(X_train.shape[0]*0.9)

    X_val = X_train[spt:]
    t_val = t_train[spt:]
    t_val = t_val.T.astype(int)

    X_train = X_train[:spt]
    t_train = t_train[:spt]
    t_train = t_train.T.astype(int)

    best_score = 0
    best_gnb = None
    prediction_accurcy = []


    # instantiate the model
    for i in range(1,times):
        gnb = sklearn.naive_bayes.GaussianNB()

        # fit the model
        gnb.fit(X_train, t_train)

        # valdiation
        t_pred = gnb.predict(X_val)
        score = sklearn.metrics.accuracy_score(t_val, t_pred)

        prediction_accurcy.append(score)        
        if (best_score<score):
            best_score =score
            best_gnb = gnb

    t_test_predict = best_gnb.predict(X_test)
    t_test_predict = t_test_predict.ravel()

    score = sklearn.metrics.accuracy_score(t_test, t_test_predict)

    coef = sklearn.inspection.permutation_importance(best_gnb, X_test, t_test_predict)

    return best_score,prediction_accurcy,coef.get('importances_mean')



def alogRandomForest(times):

    X_train = np.loadtxt("X_train.txt")
    t_train = np.loadtxt("t_train.txt")
    X_test = np.loadtxt("X_test.txt")
    t_test = np.loadtxt("t_test.txt")

    X_train, X_val, t_train, t_val = sklearn.model_selection.train_test_split(X_train, t_train, test_size = 0.1, random_state = 0)

    spt = int(X_train.shape[0]*0.9)

    X_val = X_train[spt:]
    t_val = t_train[spt:]
    t_val = t_val.T.astype(int)

    X_train = X_train[:spt]
    t_train = t_train[:spt]
    t_train = t_train.T.astype(int)

    best_score = 0
    best_dtc = None
    prediction_accurcy = []
    
    for i in range(1,times):
        #training
        rfc = sklearn.ensemble.RandomForestClassifier(max_depth=i)
        rfc.fit(X_train, t_train)
        
        # valdiation
        t_pred = rfc.predict(X_val)
        score = sklearn.metrics.accuracy_score(t_val, t_pred)

        prediction_accurcy.append(score)        
        if (best_score<score):
            best_score =score
            best_dtc = rfc

    # test
    t_test_predict = best_dtc.predict(X_test)
    score = sklearn.metrics.accuracy_score(t_test, t_test_predict)
    
    
    return best_score,prediction_accurcy,best_dtc.feature_importances_




if __name__ == "__main__":
    Accurcy=[]

    score,weightImportance= alogWQDA(10)
    Accurcy.append(str(score)+",Weighted Quadratic Discriminant Analysis(baseline)\n")
    np.savetxt("WQDAWeight.txt", weightImportance)
    print("1/6")

    bestScore,prediction_accurcy,weightImportance =  alogDecisionTree(10)
    Accurcy.append(str(bestScore)+",DecisionTree\n")
    np.savetxt("DecisionTreeWeight.txt", weightImportance)
    print("2/6")

    bestScore,prediction_accurcy,weightImportance =  alogBayesClassifier(10)
    Accurcy.append(str(bestScore)+",BayesClassifier\n")
    np.savetxt("BayesClassifierWeight.txt", weightImportance)
    print("3/6")
    
    bestScore,prediction_accurcy,weightImportance =  alogRandomForest(10)
    Accurcy.append(str(bestScore)+",RandomForest\n")
    np.savetxt("RandomForestWeight.txt", weightImportance)
    print("4/6")

    bestScore,prediction_accurcy,weightImportance =  logistic_regression.runRegreesion(10)
    Accurcy.append(str(bestScore)+",Logestic Regreesion\n")
    np.savetxt("RegreesionWeight.txt", weightImportance)
    print("5/6")


    bestScore,test_accurcy,weightImportance =  linear_regression.runRegreesion(10)
    Accurcy.append(str(bestScore)+",Linear Regreesion\n")
    np.savetxt("LinearRegreesionWeight.txt", weightImportance)
    print("6/6")


    f = open("score.txt","w")
    f.writelines(Accurcy)
    f.close()

