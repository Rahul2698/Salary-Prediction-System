import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def welcome():
    print("*** Welcome to Salary Prediction System ***")
    print("Press ENTER key to proceed")
    input()

def checkCSV():
    csv_files=[]
    current_dir=os.getcwd()
    content_list=os.listdir(current_dir)
    for x in content_list:
        if x.split('.')[-1]=="csv":
            csv_files.append(x)
    if len(csv_files)==0:
        return "No CSV file available in the directory"
    else:
        return csv_files

def display_and_select_csv(csv_files):
    i=0
    for csv_file in csv_files:
        print(i,"->",csv_file)
        i+=1
    return csv_files[int(input("Select a file to create ML Model"))]

def graph(X_train,Y_train,regressionObject,X_test,Y_test,Y_pred):
    plt.scatter(X_train,Y_train,color='red',label='Train Data')
    plt.plot(X_train,regressionObject.predict(X_train),color='blue',label='Regression Line')
    plt.scatter(X_test,Y_test,color='green',label='Test Data')
    plt.title('Salary vs Experience')
    plt.xlabel('Years of Experience')
    plt.ylabel('Salary')
    plt.legend()
    plt.show()

def main():
    welcome()
    try:
       csv_files=checkCSV()
       if csv_files=="No CSV file available in the directory":
           raise FileNotFoundError("No CSV files available in the directory")
       else:
           csv_file=display_and_select_csv(csv_files)
           print(csv_file,"is selected")
           print("Reading CSV file...")
           print("Creating DataSet")
           dataset=pd.read_csv(csv_file)
           X=dataset.iloc[:,:-1].values
           Y=dataset.iloc[:,-1].values
           s=float(input("Enter TestData Size (between 0 and 1):"))
           X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=s)
           print("Model creation is in progress...")
           regressionObject=LinearRegression()
           regressionObject.fit(X_train,Y_train)
           print("Model is Created.")
           Y_pred=regressionObject.predict(X_test)
           i=0
           print("X_TestData ... Y TestData ... Y PredData")
           while(i<len(X_test)):
               print(X_test[i],'...',Y_test[i],'...',Y_pred[i])
               i+=1
           print("Press ENTER to see above result as Graphical Representation")
           input()
           graph(X_train,Y_train,regressionObject,X_test,Y_test,Y_pred)
           r2=r2_score(Y_test,Y_pred)
           print("Our model is %2.2f%% accurate"%(r2*100))
           print("Now you can predict Salary of an Employee using our model")
           print("Please enter Experience saperated by comma(,)")
           exp=[float(e) for e in input().split(',')]
           ex=[]
           for e in exp:
               ex.append([e])
           experience=np.array(ex)
           salaries=regressionObject.predict(experience)

           plt.scatter(experience,salaries,color='brown',label='Predicted Value')
           plt.xlabel('Years of Experience')
           plt.ylabel('Salary')
           plt.legend()
           plt.show()

           d=pd.DataFrame({'Experience':exp,'Salary':salaries})
           print(d)


    except FileNotFoundError:
        print("No CSV file available in the directory")
        print("Press ENTER key to exit")
        input()
        exit()


if __name__=="__main__":
        main()
        input()
