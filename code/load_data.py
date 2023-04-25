
import numpy as np 
import pandas as pd # pandas data csv processor
import category_encoders as ce # one hot feacture for caregory varialbes

def dotDatetoDotCSV(TrainFileName,TestFileName):
    file = open(TrainFileName,"r")
    text = file.readlines()
    file.close()
    
    file = open(TrainFileName+".csv","w")
    file.writelines(text)
    file.close()


    file = open(TestFileName,"r")
    text = file.readlines()
    file.close()
    
    text.remove("|1x3 Cross validator\n")
    
    file = open(TestFileName+".csv","w")
    file.writelines(text)
    file.close()

def load_data(TrainFileName,TestFileName):

    train_df = pd.read_csv(TrainFileName, header=None, sep=',\s',engine='python')
    test_df = pd.read_csv(TestFileName, header=None, sep=',\s',engine='python')

    colNames = ["age","workclass","fnlwgt","education","education-num",
               "marital-status","occupation","relationship","race","sex",
               "capital-gain","capital-loss","hours-per-week","native-country","income_class"]
    
    train_df.columns = colNames
    test_df.columns = colNames

    # replace ? missing data to None
    categorical = [var for var in train_df.columns if train_df[var].dtype=='O']
    numerical = [var for var in train_df.columns if train_df[var].dtype!='O']

    for var in categorical: 
        train_df[var].replace('?', np.NaN, inplace=True)
        test_df[var].replace('?', np.NaN, inplace=True)

    # X ,t split
    X_train = train_df.drop(["income_class"], axis=1)
    t_train = train_df["income_class"]
    X_test = train_df.drop(["income_class"], axis=1)
    t_test = train_df["income_class"]

    # one hot encode
    X_categorical = [var for var in X_train.columns if train_df[var].dtype=='O']
    Xencoder = ce.OneHotEncoder(cols=X_categorical)
    X_train = Xencoder.fit_transform(X_train)
    X_test = Xencoder.transform(X_test)

    F = open("OneHotFeacture.txt","w")
    feacture = Xencoder.feature_names_out_
    for i in feacture:
        F.write(i+"\n")
    
    F.close()

    mapping = Xencoder.category_mapping
    F = open("cat_mapping.txt","w")
    for i in range(len(mapping)):
        F.write(str(mapping[i]))
    F.close()


    # classifiy ">50K" as 0, "<=50K" as 1
    t_train = (t_train == ">50K").astype(int)
    t_test = (t_test == ">50K").astype(int)

    np.savetxt("X_train.txt",X_train)
    np.savetxt("X_test.txt",X_test)
    np.savetxt("t_train.txt",t_train)
    np.savetxt("t_test.txt",t_test)

    return X_train,X_test,t_train,t_test,colNames

                    
if (__name__ == "__main__"):
    # load data from where orginal pull from the website
    # add a tittle line
    dotDatetoDotCSV("adult.data","adult.test")
    X_train,X_test,t_train,t_test,colNames = load_data("adult.data.csv","adult.test.csv")