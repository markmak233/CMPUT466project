import alog
import GenerateReport
import load_data
import logistic_regression
import linear_regression
import numpy as np
import time

def main():
    text = "{:<5} {:<10}"
    load_data.dotDatetoDotCSV("adult.data","adult.test")
    X_train,X_test,t_train,t_test,colNames = load_data.load_data("adult.data.csv","adult.test.csv")
    
    print(text.format("(1/9)","data classified"))

    Accurcy=[]

    N_epoch = 50

    print(text.format("(2/9)","Running Weighted Quadratic Discriminant Analysis(baseline)"))
    st = time.time()
    score,weightImportance= alog.alogWQDA(N_epoch)
    end = time.time()
    Accurcy.append(str(score)+","+str(end-st)+",Weighted Quadratic Discriminant Analysis(baseline)\n")
    np.savetxt("WQDAWeight.txt", weightImportance)


    print(text.format("(3/9)","Running Decision Tree"))
    st = time.time()
    bestScore,prediction_accurcy,weightImportance =  alog.alogDecisionTree(N_epoch)
    end = time.time()
    Accurcy.append(str(bestScore)+","+str(end-st)+",DecisionTree\n")
    np.savetxt("DecisionTreeWeight.txt", weightImportance)

    print(text.format("(4/9)","Running Bayes Classifier"))
    st = time.time()
    bestScore,prediction_accurcy,weightImportance =  alog.alogBayesClassifier(N_epoch)
    end = time.time()
    Accurcy.append(str(bestScore)+","+str(end-st)+",BayesClassifier\n")
    np.savetxt("BayesClassifierWeight.txt", weightImportance)
    
    print(text.format("(5/9)","Running Random Forest"))
    st = time.time()
    bestScore,prediction_accurcy,weightImportance =  alog.alogRandomForest(N_epoch)
    end = time.time()
    Accurcy.append(str(bestScore)+","+str(end-st)+",RandomForest\n")
    np.savetxt("RandomForestWeight.txt", weightImportance)

    print(text.format("(6/9)","Running Logestic Regression"))
    st = time.time()
    bestScore,prediction_accurcy,weightImportance =  logistic_regression.runRegreesion(N_epoch)
    end = time.time()
    Accurcy.append(str(bestScore)+","+str(end-st)+",Logestic Regreesion\n")
    np.savetxt("LogesticRegreesionWeight.txt", weightImportance)

    print(text.format("(7/9)","Running Linear Regression"))
    st = time.time()
    bestScore,test_accurcy,weightImportance =  linear_regression.runRegreesion(N_epoch)
    end = time.time()
    Accurcy.append(str(bestScore)+","+str(end-st)+",Linear Regreesion\n")
    np.savetxt("LinearRegreesionWeight.txt", weightImportance)

    f = open("score.txt","w")
    f.writelines(Accurcy)
    f.close()
    print(text.format("(8/9)","Generateing report"))
    GenerateReport.genReport()

    print(text.format("(9/9)","Rport at Report.pdf. image at AlogritmAccuracy.svg"))

if __name__ == "__main__":
    main()