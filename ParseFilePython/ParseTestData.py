import glob
import csv

indexStart = 4

path = 'Results Age Exp1/'
listFile = glob.glob(path + "*.log");
print(listFile)
stringSplit = [];
ArrayValues = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]

for i in range(len(listFile)):
    #============================== Read the filename
    stringSplit = listFile[i].split("_")
    del (stringSplit[0])
   # print(stringSplit)
    f = open(listFile[i], "r")
    # ====================get the second to last line
    lines = f.readlines()
    count = len(lines)
    data = lines[count - 2]
    data = data.split('|')

    Accuracy = data[indexStart].replace(' ','')
    AccuracyValidation = data[indexStart + 1].replace(' ','')
    Loss = data[indexStart + 2].replace(' ','')
    LossValidation = data[indexStart + 3].replace(' ','')
    print(Accuracy+" "+AccuracyValidation+" "+Loss + " "+LossValidation)
    print(ArrayValues)
    print(stringSplit[0])
    Index = int(stringSplit[0])
    ArrayValues[Index-1].append(Accuracy+","+AccuracyValidation+","+Loss+","+LossValidation)

with open("Results EXP1.csv", 'w') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL);
    filewriter.writerow(['Accuracy','Accuracy Validation','Loss','Loss Validation'])
    for i in range(len(ArrayValues)):
        filewriter.writerow(['Number Of features = ',str(i+1)]);
        for j in range(len(ArrayValues[i])):
            print(ArrayValues[i][j].split(","))
            string =  ArrayValues[i][j].split(",");
            filewriter.writerow(string)


