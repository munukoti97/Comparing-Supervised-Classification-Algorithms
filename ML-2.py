import pandas as pd
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import math 
import time
def method(l):
    # mean1 is the  avg mean of rank of SVM
 mean1=0

# mean2 is the  avg mean of rank of LogisticRegression
 mean2=0

# mean3 is the  avg mean of rank of LKNN
 mean3=0

# m is the rank list which contains the rank of each accuracy 
# assigning the rank to the each accuracy in the list "l"
# Rank-1 is for more higher  value in the row
# Rank-2  is the next higher value in the row
# Rank_-3 is the minimum value in the row
# creating 10 rows for the rank list
 for i in range(0,10):
   m.append([])
   for j in range(0,3):
       m[i].append(0)
 for i in range(0,10):
# assuming l[i][0] is the minimum value in the row 
# assign l[i][0] value to min
    min=l[i][0]
# assuming l[i][0] is the maximum value in the row
# assign l[i][0] value to max
    max=l[i][0]
    k1=0
    k=0
    m[i][0]=3
# finding the minimum and maximum value in the row 
    for j in range(0,3):
        if max<l[i][j]:
            max=l[i][j]
            m[i][j]=1
            k=j
        if min>l[i][j]:
            min=l[i][j]
            m[i][j]=3
            k1=j
# remaining element must be a middle value because one is max and another one is min only 3 elements in each row
    m[i][3-(k+k1)]=2
    mean1=mean1+m[i][0]
    mean2=mean2+m[i][1]
    mean3=mean3+m[i][2]
 mean1=mean1/10
 mean2=mean2/10
 mean3=mean3/10

# printing the table of all three algorithms   and rank of each value in the row


 print("Friedman test and results")
 print("")
 print("")
 print("Fold",end="      ")
 print("SVM",end="       ")
 print("Logistic Regression",end="    ")
 print("KNN")
 for i in range(0,10):
    if i==9:
     print(i+1,end="       ")
    else:
     print(i+1,end="        ")   
    for j in range(0,3):
        print("%.2f" % round(l[i][j],2),end=" ")
    
        print('({:1d})'.format(m[i][j]),end="         ")
    print()
 print("")    
 print("avg rank",end="    ")
 print(mean1,end="             ")
 print(mean2,end="             ")
 print(mean3)
 sum=0

 print("")
 print("")

# calculating the average rank by using the formulae = (1/n*k)*sigma of(Rij)
 for i in range(0,10):
    for j in range(0,3):
        sum=sum+m[i][j]
 average_rank_R=(sum/(3*10))
 k=average_rank_R

 print("Average rank is ",average_rank_R)
# calculating the sum of squared differences by using the formulae n*(sigma of(Rank mean - average rank)*(Rank mean - average rank))
 ss=10*((mean1-k)*(mean1-k)+(mean2-k)*(mean2-k)+(mean3-k)*(mean3-k)) 
 print("")
 print("")

 print("sum of squared differences is ",ss)
 print("")
 print("")
 if (ss>7.8):# 7.8 is the value which got from the table by using the parametrs of n=10 10 means 10 folds and k=3 3 means 3 algorithms which are used 
    
    print("reject null hypothesis all three algorithms do not perform equally and perform differently ")
 else:
    print("accept null hypothesis 3 algorithms perform same ")
    
 k=3
 n=10
# for given  α = 0.05 and k = 3  the q_alpha is 2.343.
 q_alpha=2.343
# The Nemenyi test calculates the critical difference as follows The idea is to calculate
#the critical difference (CD) against which the difference in average rank between two
# algorithms is compared
# calculate the critical difference by using the formulae q_aplha*sqrt(k*(k+1)/6*n)
 cd=q_alpha*(math.sqrt( (k*(k+1))/(6*n) ) )

 print("")
 print("")
 print("Critical difference for Nemenyi test is ",cd)

 print("")
 if abs(mean1-mean3)>cd:
   print("SVM and KNN perform algorithms differently exceeds critical difference ")
 if abs(mean2-mean1)>cd:
   print("Logistic Regression and SVM perform algorithms differently exceeds critical difference ") 
 if abs(mean2-mean3)>cd:
   print("KNN and Logistic Regression perform algorithms differently exceed critical difference") 
    
 print("")    
 print("************************************************************************************************")

def method1(l):
    # mean1 is the  avg mean of rank of SVM
 mean1=0

# mean2 is the  avg mean of rank of LogisticRegression
 mean2=0

# mean3 is the  avg mean of rank of LKNN
 mean3=0

# m is the rank list which contains the rank of each accuracy 
# assigning the rank to the each accuracy in the list "l"
# Rank-1 is for minimum time value in the row
# Rank-2  is the next higher value in the row
# Rank_-3 is the maximum time value in the row
# creating 10 rows for the rank list
 for i in range(0,10):
   m.append([])
   for j in range(0,3):
       m[i].append(0)
 for i in range(0,10):
# assuming l[i][0] is the minimum value in the row 
# assign l[i][0] value to min
    min=l[i][0]
# assuming l[i][0] is the maximum value in the row
# assign l[i][0] value to max
    max=l[i][1]
    k1=0
    k=0
    m[i][0]=1
    m[i][1]=3
# finding the minimum and maximum value in the row 
    for j in range(0,3):
        if max<l[i][j]:
            max=l[i][j]
            m[i][j]=3
            k=j
        if min>l[i][j]:
            min=l[i][j]
            m[i][j]=1
            k1=j
# remaining element must be a middle value because one is max and another one is min only 3 elements in each row
    m[i][3-(k+k1)]=2
    mean1=mean1+m[i][0]
    mean2=mean2+m[i][1]
    mean3=mean3+m[i][2]
 mean1=mean1/10
 mean2=mean2/10
 mean3=mean3/10

# printing the table of all three algorithms   and rank of each acccuracy in the row


 print("Friedman test and results")
 print("")
 print("")
 print("Fold",end="      ")
 print("SVM",end="       ")
 print("Logistic Regression",end="    ")
 print("KNN")
 for i in range(0,10):
    if i==9:
     print(i+1,end="       ")
    else:
     print(i+1,end="        ")   
    for j in range(0,3):
        print("%.2f" % round(l[i][j],2),end=" ")
    
        print('({:1d})'.format(m[i][j]),end="         ")
    print()
 print("")    
 print("avg rank",end="    ")
 print(mean1,end="             ")
 print(mean2,end="             ")
 print(mean3)
 sum=0

 print("")
 print("")

# calculating the average rank by using the formulae = (1/n*k)*sigma of(Rij)
 for i in range(0,10):
    for j in range(0,3):
        sum=sum+m[i][j]
 average_rank_R=(sum/(3*10))
 k=average_rank_R

 print("Average rank is ",average_rank_R)
# calculating the sum of squared differences by using the formulae n*(sigma of(Rank mean - average rank)*(Rank mean - average rank))
 ss=10*((mean1-k)*(mean1-k)+(mean2-k)*(mean2-k)+(mean3-k)*(mean3-k)) 
 print("")
 print("")

 print("sum of squared differences is ",ss)
 print("")
 print("")
 if (ss>7.8):# 7.8 is the value which got from the table by using the parametrs of n=10 10 means 10 folds and k=3 3 means 3 algorithms which are used 
    
    print("reject null hypothesis all three algorithms do not perform equally and perform differently ")
 else:
    print("accept null hypothesis 3 algorithms perform same ")
    
 k=3
 n=10
# for given  α = 0.05 and k = 3  the q_alpha is 2.343.
 q_alpha=2.343
# The Nemenyi test calculates the critical difference as follows The idea is to calculate
#the critical difference (CD) against which the difference in average rank between two
# algorithms is compared
# calculate the critical difference by using the formulae q_aplha*sqrt(k*(k+1)/6*n)
 cd=q_alpha*(math.sqrt( (k*(k+1))/(6*n) ) )

 print("")
 print("")
 print("Critical difference for Nemenyi test is ",cd)

 print("")
 if abs(mean1-mean3)>cd:
   print("SVM and KNN perform algorithms differently exceeds critical difference ")
 if abs(mean2-mean1)>cd:
   print("Logistic Regression and SVM perform algorithms differently exceeds critical difference ") 
 if abs(mean2-mean3)>cd:
   print("KNN and Logistic Regression perform algorithms differently exceed critical difference") 
    
 print("")    
 print("************************************************************************************************")
 
def printtables(l,mean1,mean2,mean3):
 print("")   
    
 print("")
 print("")
 print("Fold",end="      ")
 print("SVM",end="       ")
 print("Logistic Regression",end="    ")
 print("KNN")
 for i in range(0,10):
    if i==9:
     print(i+1,end="       ")
    else:
     print(i+1,end="        ")   
    for j in range(0,3):
        print("%.2f" % round(l[i][j],2),end="             ")
    
        
    print()
 print("")    
 print("avg mean",end=" ")
 print("%.2f" % round(mean1,2),end="             ")
 print("%.2f" % round(mean2,2),end="             ")
 print("%.2f" % round(mean3,2))
 print("")
 print("")





dataset = pd.read_csv('spambase.data')
# taking the data into list
dataset_list = dataset.values.tolist() 

# taking all the columns and rows except the last column which is spam or not spam
dataset_input_list = dataset.iloc[:,:-1].values.tolist()

#  taking all the columns and rows only the column which is spam or not spam(0 or 1)
dataset_output_list = dataset.iloc[:,-1].values.tolist()

# the module which does split the data into folds
from sklearn.model_selection import StratifiedKFold

# here our requirement is 10 split the data into 10 folds with equally spliting and equally spreading the values to testing data & ttraining data 

skf = StratifiedKFold(n_splits=10)

# here 
skf.get_n_splits(dataset_input_list,dataset_output_list)
StratifiedKFold(n_splits=10, random_state=None, shuffle=False)

j=0
# here l is the 2d  list which contains accuracy of LogisticRegression,SVM and KNN
# l is the predictive performance
l=[]

# Time perform is the list which contains the time in secs  time taken by each algorithm to execute the each fold of data
# timeperform is the computational performance (training time)
timeperform=[]
# here recall is the 2d  list which contains f -measure recall accuracy of LogisticRegression,SVM and KNN
recall=[]
# here precision is the 2d  list which contains f -measure precision accuracy of LogisticRegression,SVM and KNN
precision=[]
# here f_measure is the 2d  list which contains calculated value of (2*precision *recall)/(precision + recall)
f_measure=[]
# mean1 is the  avg mean of accuracy of LogisticRegression
mean1=0

# mean2 is the avg mean of  accuracy of SVM
mean2=0

# mean3 is the avg mean of  accuracy of KNN
mean3=0

#mean1_trainingtime_svm  is the  avg mean of computational performance (training time) of SVM
mean1_trainingtime_svm=0
#mean1_trainingtime_lgr  is the  avg mean of computational performance (training time) of LogisticRegression
mean1_trainingtime_lgr=0
#mean1_trainingtime_knn  is the  avg mean of computational performance (training time) of knn
mean1_trainingtime_knn=0

# mean_f_svm_recall is the avg mean of predictive performance of f-measure recall of SVM
mean_f_svm_recall=0
# mean_f_lgr_recall is the avg mean of predictive performance of f-measure recall of LogisticRegression
mean_f_lgr_recall=0


# mean_f_knn_recall is the avg mean of predictive performance of f-measure recall of knn
mean_f_knn_recall=0

#mean_f_svm_precision is the avg mean of predictive performance of f-measure precision of SVM
mean_f_svm_precision=0
#mean_f_lgr_precision is the avg mean of predictive performance of f-measure precision of LogisticRegression
mean_f_lgr_precision=0
#mean_f_knn_precision is the avg mean of predictive performance of f-measure precision of knn
mean_f_knn_precision=0

# mean_f_measure_svm is the avg mean of predictive performance of f-measure  of SVM
mean_f_measure_svm=0
# mean_f_measure_lgr is the avg mean of predictive performance of f-measure  of LogisticRegression
mean_f_measure_lgr=0
# mean_f_measure_knn is the avg mean of predictive performance of f-measure  of knn
mean_f_measure_knn=0

# here creating the list ,recall&precision ,f_measure, timeperform into 10 rows and 3 columns  because our dividing the data into 10 folds 
for i in range(0,10):
   l.append([]) 
   timeperform.append([])
   recall.append([])
   precision.append([])
   f_measure.append([])
# calculating accuracy,time taken by ecah classifier , f value store this values into List,Time perform in every iteration
for train_index, test_index in skf.split(dataset_input_list,dataset_output_list):
    
   # train_index_list = train_index.tolist()
# a is the list of all the data from training data which contains  columns and rows except the last column which is spam or not spam
    a=[]
# b is the list of all the data from training data  taking all the columns and rows only the column which is spam or not spam(0 or 1)
    b=[]
#c is the list of all the data from testing data which contains  columns and rows except the last column which is spam or not spam
    c=[]
# d is the list of all the data from testing  data  taking all the columns and rows only the column which is spam or not spam(0 or 1)
    d=[]

# train_index contains the list of numbers which are 90% of random values   the range of values are from 0- 4600
    for i in train_index:
       
        a.append(dataset_list[i][:-1])
        b.append(dataset_list[i][-1])
        
# test index contains the 10% of numbers  remaining will be 90%  of train_index   
    for i in test_index:
     
        c.append(dataset_list[i][:-1])
        d.append(dataset_list[i][-1])
# time.clock() is the start time before algorithm starts    
       
    svmclf = svm.SVC(gamma = 'scale')
    start_time=time.time() 
    svmclf.fit(a,b)
# by using the testing data svmresult_list predicates whether it is spam or not spam 
    svmresult_list = svmclf.predict(c)
# time end is the how much time has been taken to perform svm algorithm 
    time_end=time.time()-start_time
    
# store the time value in time_perform list
    timeperform[j].append(time_end)
    
    start_time=time.time()
    lgsregression = LogisticRegression(random_state=0,solver='lbfgs').fit(a,b)
# by using the testing data lgsregression_result_list predicates whether it is spam or not spam 
    lgsregression_result = lgsregression.predict(c)
# time end is the how much time has been taken to perform   LogisticRegression algorithm
    time_end=time.time()-start_time

# store the time value in time_perform list
    timeperform[j].append(time_end)
    
    
    knn = KNeighborsClassifier(n_neighbors = 5)
    start_time=time.time()
    knn.fit(a,b)    
    knn_result = knn.predict(c)
    time_end=time.time()-start_time
    timeperform[j].append(time_end)
# confusion matrix which contains true postive ,true negative,false postive,false negative    
    tn_svm, fp_svm, fn_svm, tp_svm = confusion_matrix(d,svmresult_list).ravel()
    #print('SVM accuracy')
# accuracy of SVM the value will be stored into list l
    #print((tp_svm+tn_svm)/(tn_svm + tp_svm + fn_svm + fp_svm) * 100)
    l[j].append((tp_svm+tn_svm)/(tn_svm + tp_svm + fn_svm + fp_svm))
    recall[j].append((tp_svm)/(tp_svm+fn_svm))
    precision[j].append((tp_svm)/(tp_svm+fp_svm))
    tn_lgs, fp_lgs, fn_lgs, tp_lgs = confusion_matrix(d,lgsregression_result).ravel()
    #print('logistic regression accuracy')
# accuracy of logistic regression  the value will be stored into list l
   # print((tp_lgs + tn_lgs)/(tn_lgs + tp_lgs + fn_lgs + fp_lgs) * 100)
    l[j].append((tp_lgs + tn_lgs)/(tn_lgs + tp_lgs + fn_lgs + fp_lgs))
    recall[j].append((tp_lgs)/(tp_lgs+fn_lgs))
    precision[j].append((tp_lgs)/(tp_lgs+fp_lgs))
    
    
    tn_knn, fp_knn, fn_knn, tp_knn = confusion_matrix(d,knn_result).ravel()
    
# accuracy of  knn   the value will be stored into list l
    #print((tp_knn + tn_knn)/(tn_knn + tp_knn + fn_knn + fp_knn) * 100)
    l[j].append((tp_knn + tn_knn)/(tn_knn + tp_knn + fn_knn + fp_knn))
    recall[j].append((tp_knn)/(tp_knn+fn_knn))
    precision[j].append((tp_knn)/(tp_knn+fp_knn))    
    
# calculating the f_measure using the formulae of (2*precision *recall)/(precision + recall)
    f_measure[j].append((2*recall[j][0]*precision[j][0])/(recall[j][0]+precision[j][0]))
    f_measure[j].append((2*recall[j][1]*precision[j][1])/(recall[j][1]+precision[j][1]))
    f_measure[j].append((2*recall[j][2]*precision[j][2])/(recall[j][2]+precision[j][2]))
    mean1=mean1+l[j][0]
    mean2=mean2+l[j][1]
    mean3=mean3+l[j][2]
    
    mean_f_svm_recall=mean_f_svm_recall+recall[j][0]
    mean_f_lgr_recall=mean_f_lgr_recall+recall[j][1]
    mean_f_knn_recall=mean_f_knn_recall+recall[j][2]
    
    mean1_trainingtime_svm=mean1_trainingtime_svm+timeperform[j][0]
    mean1_trainingtime_lgr=mean1_trainingtime_lgr+timeperform[j][1]
    mean1_trainingtime_knn=mean1_trainingtime_knn+timeperform[j][2]
    
    mean_f_svm_precision=mean_f_svm_precision+precision[j][0]
    mean_f_lgr_precision=mean_f_lgr_precision+precision[j][1]
    mean_f_knn_precision=mean_f_knn_precision+precision[j][2]
    
    mean_f_measure_svm=mean_f_measure_svm+f_measure[j][0]
    mean_f_measure_lgr=mean_f_measure_svm+f_measure[j][1]
    mean_f_measure_knn=mean_f_measure_knn+f_measure[j][2]
    j=j+1
m=[]
mean1=mean1/10
mean2=mean2/10
mean3=mean3/10

mean_f_svm_recall=mean_f_svm_recall/10
mean_f_lgr_recall=mean_f_lgr_recall/10
mean_f_knn_recall=mean_f_knn_recall/10

mean_f_svm_precision=mean_f_svm_precision/10
mean_f_lgr_precision=mean_f_lgr_precision/10
mean_f_knn_precision=mean_f_knn_precision/10


mean1_trainingtime_svm=mean1_trainingtime_svm/10
mean1_trainingtime_lgr=mean1_trainingtime_lgr/10
mean1_trainingtime_knn=mean1_trainingtime_knn/10


mean_f_measure_svm=mean_f_measure_svm/10
mean_f_measure_lgr=mean_f_measure_svm/10
mean_f_measure_knn=mean_f_measure_knn/10


print("")
print("")
print("")
print("")
# printing the table of all three algorithms accuracy  and time performance of each algorithm in every fold


print("")
print("")
# printing the table of all three algorithms accuracy   of each algorithm in every fold
print("Table for predictive performance of three algorithms accuracy ")

printtables(l,mean1,mean2,mean3)

# printing the table of all three algorithms computational performance (training time) of each algorithm in every fold
print("table for training time")

print("table of all three algorithms computational performance (training time) of each algorithm in every fold ")
printtables(timeperform,mean1_trainingtime_svm,mean1_trainingtime_lgr,mean1_trainingtime_knn)

# printing the table of all three algorithms predictive performance of recall measure of each algorithm in every fold
print("table for f measure recall")
print("table of all three algorithms predictive performance of recall measure of each algorithm in every fold")
printtables(recall,mean_f_svm_recall,mean_f_lgr_recall,mean_f_knn_recall)

# printing the table of all three algorithms predictive performance of precision measure of each algorithm in every fold
print("table for f measure precision")
print("table of all three algorithms predictive performance of precision measure of each algorithm in every fold")
printtables(precision,mean_f_svm_precision,mean_f_lgr_precision,mean_f_knn_precision)

# printing the table of all three algorithms predictive performance of f- measure of each algorithm in every fold
print("table for f measure ")
print("table of all three algorithms predictive performance of f- measure of each algorithm in every fold")
printtables(f_measure,mean_f_measure_svm,mean_f_measure_lgr,mean_f_measure_knn)
print("************************************************************************************************")
# method(l) what it does is printing the table of all three algorithms accuracy  and rank of each acccuracy in the row
print("Friedman test and results for predictive performance of  3 algorithms accuracy  ")
print("perform test on predictive performance of  3 algorithms accuracy  ")
# calling method to conduct Friedman test & Nemeyi test on predictive performance of  3 algorithms accuracy
method(l)
print("************************************************************************************************")
# method1(timeperform) what it does is printing the table of all three algorithms computational performance (training time)  and rank of each acccuracy in the row
print("Friedman test and results for computational performance (training time) of  3 algorithms   ")
print("perform test on computational performance (training time) of  3 algorithms  ")
# calling method to conduct Friedman test & Nemeyi test on computational performance (training time) of  3 algorithms 
method1(timeperform)
print("************************************************************************************************")
# method(recall) what it does is printing the table of all three algorithms predictive performance  of f-measure for recall  and rank of each acccuracy in the row
print("Friedman test and results for predictive performance   for recall   of  3 algorithms ")
print("perform test on predictive performance   for recall   of  3 algorithms ")
# calling method to  conduct Friedman test & Nemeyi test on predictive performance  of f-measure for recall   of  3 algorithms
method(recall)
print("************************************************************************************************")
#method(recall) what it does is printing the table of all three algorithms predictive performance  of f-measure for precision  and rank of each acccuracy in the row
print("Friedman test and results for predictive performance  for  precision   of  3 algorithms ")
print("perform test on predictive performance   for precision   of  3 algorithms ")
# calling method to  conduct Friedman test & Nemeyi test on predictive performance   for precision  of  3 algorithms
method(precision)
print("************************************************************************************************")
#method(f_measure) what it does is printing the table of all three algorithms predictive performance  of f-measure   and rank of each acccuracy in the row
print("Friedman test and results for predictive performance  of f-measure    of  3 algorithms ")
print("perform test on predictive performance  of f-measure   of  3 algorithms ")
# calling method to  conduct Friedman test & Nemeyi test on predictive performance  of f-measure  
method(f_measure)
print("************************************************************************************************")
