import numpy as np

# this file generate the report

def Top10importWeight(alog, feacture, weight,num):
    # absolute weight
    ot=[]
    ot.append("\n\n\n")
    ot.append("Alogrithm: "+alog+"\n")
    ot.append("========================\n")
    absed_weight = np.absolute(weight).tolist()
    for i in range(num):
        w_m =  max(absed_weight)
        text = "{:<3} {:<20} {:<15}\n"
        ot.append(text.format(i+1,feacture[absed_weight.index(w_m)],str(w_m)))
        feacture.remove(feacture[absed_weight.index(w_m)])
        absed_weight.remove(w_m)

    f = open("ProgrammeOutput.txt","a")
    f.writelines(ot)
    f.close()

        

def genReport():
    ot = ["Report\n\n"]

    f = open("score.txt","r")
    text = f.readlines()
    f.close()

    
    dummy_per = None
    acc_dict = {}
    for i in range(len(text)):
        acc = text[i][:-1].split(",")
        time = "{:<5} {:<30}"
        acc_dict[acc[0]] = time.format(float(acc[1]),acc[2])
        
    dummy_per = min(acc_dict.keys())

    ot.append("Alogrithm Accuracy,time,name\n")
    ot.append("========================\n")
    while len(acc_dict) != 0:
        per = max(acc_dict.keys())
        ot.append(str(str(per)+" "+acc_dict[per]+"\n"))
        acc_dict.pop(per)


    f = open("ProgrammeOutput.txt","w")
    f.writelines(ot)
    f.close()

    f = open("OneHotFeacture.txt","r")
    text = f.readlines()
    f.close()

    top = 10
    feacture=[]
    for i in range(len(text)):
        feacture.append(text[i][:-1])
    weight = np.loadtxt("DecisionTreeWeight.txt")
    Top10importWeight("DecisionTree",feacture,weight,top)
    feacture.clear()

    feacture=[]
    for i in range(len(text)):
        feacture.append(text[i][:-1])
    weight = np.loadtxt("BayesClassifierWeight.txt")
    Top10importWeight("BayesClassifier",feacture,weight,top)
    feacture.clear()

    feacture=[]
    for i in range(len(text)):
        feacture.append(text[i][:-1])
    weight = np.loadtxt("RandomForestWeight.txt")
    Top10importWeight("Random Forest",feacture,weight,top)
    feacture.clear()

    feacture=[]
    for i in range(len(text)):
        feacture.append(text[i][:-1])
    weight = np.loadtxt("LogesticRegreesionWeight.txt")
    Top10importWeight("Logestic Regreesion",feacture,weight,top)

    feacture=[]
    for i in range(len(text)):
        feacture.append(text[i][:-1])
    weight = np.loadtxt("LinearRegreesionWeight.txt")
    Top10importWeight("Linear Regreesion",feacture,weight,top)

    feacture=[]
    for i in range(len(text)):
        feacture.append(text[i][:-1])
    weight = np.loadtxt("WQDAWeight.txt")
    Top10importWeight("Weighted Quadratic Discriminant Analysis(base line)",feacture,weight,top)


    F = open("cat_mapping.txt","r")
    t = F.readlines()
    F.close()

    f = open("ProgrammeOutput.txt","a")
    f.write("\ncatelogy reference:\n")
    f.writelines(t)
    f.close()





if __name__ == "__main__":
    genReport()