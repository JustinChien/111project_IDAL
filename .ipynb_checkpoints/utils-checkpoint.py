import os
import matplotlib.pyplot as plt
import itertools
import prettytable as pt
import pandas as pd
import numpy as np


report_csv_path = "D:\\111project\\gitHub\\041-111project\\data\\history_report_CalcOnly.csv"


def join_path(root:str , *params)-> str:
    # give root and folders / files will return a path in str
    temp = root
    for p in params:
        temp = os.path.join(temp,p)
    return temp

#This will plot a binary or multiclass classification's confusion matrix
def plot_confusion_matrix(cm, classes
                          ,normalize = False
                          ,title='Confusion matrix'
                          ,cmap=plt.cm.Blues):
    
    plt.figure(figsize=(6,6))
    plt.imshow(cm,interpolation='nearest',cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks,classes,rotation=45,fontsize=12)
    plt.yticks(tick_marks,classes,fontsize=12)
    if normalize :
        cm = cm.astype('float') / cm.sum(axis=1)[:,np/newaxis]
        # print("Normalized confusion matrix")
    else :
        # print("Confusion matrix")
        pass
    # print(cm)
    
    thresh = cm.max() / 2.
    for i , j in itertools.product(range(cm.shape[0]),range(cm.shape[1])): 
        plt.text(j,i,cm[i,j],
                ha = "center",
                fontsize = 15,
                color = "orange" if cm[i,j]> thresh else "purple")
        
    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

def plot_multi_label_confusion_matrix(cm,classes,cmap=plt.cm.Blues) -> None:
    #give the whole ConfusionMatrix and classes and it will plot it out
    fig,ax = plt.subplots(2,int(len(cm)/2))
    count = 0
    for a,b in itertools.product(range(2),range(int(len(cm)/2))):


        thresh = cm[count].max() / 2.
        temp_cm = cm[count]
        class_ = classes[count]

        # plt.figure(figsize=(6,6))
        ax[a][b].imshow(temp_cm,interpolation='nearest',cmap=plt.cm.Blues)
        ax[a][b].set_title(class_)

        if(a == 0 and b==0):
            ax[a][b].set_yticks(np.arange(2),["TrueNegative , FalsePositive","FalseNegative , TruePositive "])
        else:
            ax[a][b].set_yticks([])

        ax[a][b].set_xticks([])
        for i,j in itertools.product(range(len(temp_cm[:][0])),range(len(temp_cm[0][:]))):
                ax[a][b].text(j,i,temp_cm[i][j],
                        ha = "center",
                        color = "orange" if temp_cm[i,j]> thresh else "purple")
        count += 1

def plot_acc_loss(acc,val_acc,loss,val_loss):
    #give values from fit history and it will plot out a graph
    plt.figure(figsize=(8,8))

    plt.subplot(2,1,1)
    plt.plot(acc,label="Trainging Accuracy")
    plt.plot(val_acc,label="Validation Accuracy")
    plt.legend(loc="upper left")
    plt.ylabel("Accuracy")
    plt.ylim([min(plt.ylim()),1])
    plt.title("Training and Validation Accuracy")

    plt.subplot(2,1,2)
    plt.plot(loss,label="Trainging Loss")
    plt.plot(val_loss,label="Validation Loss")
    plt.legend(loc="upper right")
    plt.ylabel("Cross Entropy")
    # plt.ylim([0,1.0]) #too high so let's not use this now
    plt.title("Trainging and Validation Loss")
    plt.xlabel("epoch")
    plt.show()

def plot_imgs(lots_imgs,lots_labels):
    # give 9 imgs and 9 lables,it will plot it out
    imgs = lots_imgs[:9]
    labels = lots_labels[:9]
    
    plt.figure(figsize=(15,15))
    for i in range(len(imgs)):
        ax = plt.subplot(3,3,i+1)
        plt.imshow(imgs[i])
        plt.title(labels[i])
        plt.axis("off")

def convert_cm_to_container(cm):
    #give it a Multilabel Confusion Matrix and it will store it in a str, ez for storing in csv
    cm_container = str()
    if len(cm)==2:
        for i ,j in itertools.product(range(len(cm)),range(2)):
            cm_container += "{}_".format(cm[i,j])
    else:
        for i,j,k in itertools.product(range(len(cm)),range(2),range(2)):
            cm_container += "{}_".format(cm[i,j,k])
    return cm_container

def convert_container_to_cm(cm_container):
    #give it the str generate from function above,will return what it suppose to look like originally
    temp = cm_container.split(sep="_")
    size = int((len(temp)-1)/4)
    # print(size)
    if size == 4 or size == 6:
        c_cm = np.empty((size,2,2))
        count = -1
        for i,j,k in itertools.product(range(size),range(2),range(2)):
            count += 1
            # print(count)
            c_cm[i,j,k] = temp[count]
    elif size == 1:
        c_cm = np.empty((2,2))
        count = -1
        for i,j in itertools.product(range(2),range(2)):
            count += 1
            # print(count)
            c_cm[i,j] = temp[count]
    return c_cm

def convert_list_to_str(list_:list) -> str:
    temp = str()
    for item in list_:
        temp += "{}_".format(item)
    return temp

def convert_str_to_list(str_:str) -> list:
    list_ = list()
    str_ = str_.split(sep='_')
    for item in str_[:-1]:
        list_.append(float(item))
    return list_

def show_report(*nums,show_plot:bool=False):
    #give nums to choose which data to show,CAN give multiplay nums at once
    #if want to print out all history better use for loop with it
    #show_plot will control whether to plot ConfusionMatrix or not
    temp = list()
    history_report = pd.read_csv(report_csv_path)

    table = pt.PrettyTable()
    table.set_style(pt.DOUBLE_BORDER)
    table.hrules=pt.ALL
    temp.append('index')
    for num in nums:
        temp.append(num)
    table.field_names = temp

    for column in ["Date","Model","Batch_size","epochs","Learning_rate","Training_Report","DATA_TYPE","Note"]:
        temp = []
        temp.append(column)
        for num in nums:
            temp.append(history_report.iloc[num][column])
        table.add_row(temp)
    print(table)

    if show_plot == True:
        for num in nums:
            current_cm = convert_container_to_cm(history_report.iloc[num]['Confusion_Matrix'])
            # print(current_cm)
            if len(current_cm)==6:
                plot_multi_label_confusion_matrix(cm=current_cm,classes=['Benign', 'CC', 'Calc', 'MLO', 'Mass', 'Melignant'])
            elif len(current_cm)==4:
                plot_multi_label_confusion_matrix(cm=current_cm,classes=['Benign', 'Calc', 'Mass', 'Melignant'])
            elif len(current_cm)==2:
                plot_confusion_matrix(cm=current_cm,classes=['Benign','Malignant'])
                
            current_row = history_report.iloc[num]
            if current_row.isnull()['acc']==True:
                print('index:{} have no acuracy data for plotting accuracy.'.format(num))
                continue
            print("index:{}".format(num))
            acc = convert_str_to_list(current_row['acc'])
            val_acc = convert_str_to_list(current_row['val_acc'])
            loss = convert_str_to_list(current_row['loss'])
            val_loss = convert_str_to_list(current_row['val_loss'])
            # print(acc,'\n',val_acc,'\n',loss,'\n',val_loss)
            plot_acc_loss(acc=acc,val_acc=val_acc,loss=loss,val_loss=val_loss)
                
            