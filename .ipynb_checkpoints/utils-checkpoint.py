import os
import matplotlib.pyplot as plt
import itertools
import prettytable as pt
import pandas as pd
import numpy as np

import config

def join_path(root:str , *params)-> str:
    
    # root    : the root path for the join
    # *params : a list of file and folders to add after the root path
    
    temp = root
    for p in params:
        temp = os.path.join(temp,p)
    return temp

def plot_confusion_matrix(cm, classes
                          ,normalize = False
                          ,title='Confusion matrix'
                          ,cmap=plt.cm.Blues):
    # plot_confusion_matrix will plot a binary or multiclass classification's confusion matrix
    
    # cm        : confusion_matrix from sklearn.Confusion_Matrix
    # classes   : the classes of cm
    # normalize : whether to normalize the number
    # title     : title of the plot
    # cmap      : the heatmap color

    
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
    
    # plot_multi_label_confusion_matrix will plot a multilabel classification's confusion matrix
    
    # cm        : confusion_matrix from sklearn.Confusion_Matrix
    # classes   : the classes of cm
    # cmap      : the heatmap color
    
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
    
    # plot out the line chart of those values
    
    # acc : the accuracy from history
    # val_acc : the validation_accuracy from history
    # loss : the loss from history
    # val_loss : the validation_loss from history
    
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
    
    # will plot the first 9 imgs
    
    # lots_imgs : the set of imgs you want to plot
    # lots_labels : the labels from the imgs you want to plot
    
    imgs = lots_imgs[:9]
    labels = lots_labels[:9]
    
    plt.figure(figsize=(15,15))
    for i in range(len(imgs)):
        ax = plt.subplot(3,3,i+1)
        plt.imshow(imgs[i])
        plt.title(labels[i])
        plt.axis("off")

def convert_cm_to_container(cm):

    # shape a confusion matrix into a str of value seperate with "_" 
    # cm : the confusion matrix you want to convert
    
    cm_container = str()
    if len(cm)==2:
        for i ,j in itertools.product(range(len(cm)),range(2)):
            cm_container += "{}_".format(cm[i,j])
    else:
        for i,j,k in itertools.product(range(len(cm)),range(2),range(2)):
            cm_container += "{}_".format(cm[i,j,k])
    return cm_container

def convert_container_to_cm(cm_container):
    
    # convert a str from convert_cm_to_container() and reshape it back to the confusion matrix
    # cm_container : the str from convert_cm_to_container()
    
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
    
    # convert a list into a str of values seperate with "_"
    # list : the list you want to convert
    
    temp = str()
    for item in list_:
        temp += "{}_".format(item)
    return temp

def convert_str_to_list(str_:str) -> list:
    
    # convert the str from convert_list_to_str() back to list
    # str_ : the string from convert_list_to_str()
    
    list_ = list()
    str_ = str_.split(sep='_')
    for item in str_[:-1]:
        list_.append(float(item))
    return list_

def show_report(*nums,report_csv_path=config.report_csv_path,show_plot:bool=False):
    
    # it will read the report_csv and print out the data you choose
    # params with "*" will read from config.py if not given
    
    # nums : the index of the data you want to print out
    # * report_csv_path : the csv file's path
    # * show_plot : whether to show the data's plot
    
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
                
            