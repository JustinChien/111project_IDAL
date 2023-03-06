from tensorflow.keras.preprocessing import image
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from tqdm import tqdm
import random

import utils
import config

X_dataset = list()


def loading_imgs(IMG_SIZE=config.IMG_SIZE,data_type=config.data_type,show_detail=config.show_detail):
    
    # Beaware that data_type's options : ALL,CC,MLO  are refer to the Calc part of cropped image dataset
    # And the options : FULL_ALL,FULL_CC,FULL_MLO are refer to the Calc part of full image of CBIS-DDSM
    # option : CatnDog will load the dataset of cats and dogs as an example dataset for testing.
    # params with "*" will read from config.py if not given
    
    # * data_type   : choose what data to load , options:ALL,CC,MLO,FULL_ALL,FULL_CC,FULL_MLO,CatnDog
    # * IMG_SIZE    : the img_size you'll use for model
    # * show_detail : whether to print out the info

    IMG_DIR = config.IMG_DIR[data_type]
    if data_type == 'DognCat':

        df = pd.read_csv(config.data_csv_path[data_type])
        for i in tqdm(range(df.shape[0])):
            img = image.load_img(utils.join_path(IMG_DIR ,df['path'][i]),target_size=IMG_SIZE)
            img = image.img_to_array(img)
            X_dataset.append(img)

        X = np.array(X_dataset)
        y = np.array(df.drop(['path'],axis=1))

        # split the data into 70% as training set and 30% as testing set 
        X_train, X_test ,y_train , y_test = train_test_split(X,y,random_state=config.R_SEED,test_size=0.3,shuffle = True)
        
    else:
        #Prepare data for feeding into the model 
        df = pd.read_csv(config.data_csv_path[data_type]) #['Benign', 'CC', 'Calc', 'MLO', 'Mass', 'Melignant']

        for i in tqdm(range(df.shape[0])):
            img = image.load_img(utils.join_path(IMG_DIR ,df['img_name'][i]),target_size=IMG_SIZE)
            img = image.img_to_array(img)
            # img = img/255. #use keras layers to preprocess imgs
            X_dataset.append(img)

        X = np.array(X_dataset)
        y = np.array(df.drop(['img_name','Calc/Mass','R/L', 'CC/MLO', 'Benign/Malignant'],axis=1))


        # split the data into 70% as training set and 30% as testing set 
        X_train, X_test ,y_train , y_test = train_test_split(X,y,random_state=config.R_SEED,test_size=0.3,shuffle = True)
        
        # test if data import correctly
        if show_detail:
            imgs = X_train[0:9]
            labels = y_train[0:9]
            imgs = imgs/255.
            utils.plot_imgs(imgs,labels)

            imgs_1 = X_test[0:9]
            labels_1 = y_test[0:9]
            imgs_1 = imgs_1/255.
            utils.plot_imgs(imgs_1,labels_1)
        
        return X_train, X_test ,y_train , y_test


def under_sampling(IMG_SIZE=config.IMG_SIZE,data_type=config.data_type,show_detail=config.show_detail):
    
    # This function will do everything loading_imgs() do, BUT will randomly choose Benign's data so that 
    # we'll have a relatively balanced dataset which Benign and Malignant have the same amout of data. 
    # Be Aware that data_type's options : ALL,CC,MLO  are refer to the Calc part of cropped image dataset
    # And the options : FULL_ALL,FULL_CC,FULL_MLO are refer to the Calc part of full image of CBIS-DDSM
    # option : CatnDog will load the dataset of cats and dogs as an example dataset for testing.
    # params with "*" will read from config.py if not given
    
    # * data_type   : choose what data to load , options:ALL,CC,MLO,FULL_ALL,FULL_CC,FULL_MLO,CatnDog
    # * IMG_SIZE    : the img_size you'll use for model
    # * show_detail : whether to print out the info
    
    IMG_DIR = config.IMG_DIR[data_type]
    Benign = list()
    Malignant = list()
    
    df = pd.read_csv(config.data_csv_path[data_type]) #['Benign', 'CC', 'Calc', 'MLO', 'Mass', 'Melignant']
    
    for index in range(len(df)):
        if df.iloc[index]['Benign/Malignant'] == 'BENIGN':
            Benign.append(df.iloc[index])
        elif df.iloc[index]['Benign/Malignant'] == 'MALIGNANT':
            Malignant.append(df.iloc[index])
            
    less_Benign = random.sample(Benign,k=len(Malignant))
    less_Benign = pd.DataFrame(less_Benign)
    Malignant = pd.DataFrame(Malignant)

    temp = pd.concat([less_Benign,Malignant])
    temp = temp.reset_index()
    
    for i in tqdm(range(temp.shape[0])):
            img = image.load_img(utils.join_path(IMG_DIR ,temp['img_name'][i]),target_size=IMG_SIZE)
            img = image.img_to_array(img)
            # img = img/255. #use keras layers to preprocess imgs
            X_dataset.append(img)
    
    X = np.array(X_dataset)
    y = np.array(temp.drop(['img_name','Calc/Mass','R/L', 'CC/MLO', 'Benign/Malignant','index'],axis=1))
    
    X_train, X_test ,y_train , y_test = train_test_split(X,y,random_state=config.R_SEED,test_size=0.3,shuffle = True)
    
    if show_detail:
        imgs = X_train[0:9]
        labels = y_train[0:9]
        imgs = imgs/255.
        utils.plot_imgs(imgs,labels)

        imgs_1 = X_test[0:9]
        labels_1 = y_test[0:9]
        imgs_1 = imgs_1/255.
        utils.plot_imgs(imgs_1,labels_1)
        
    return X_train, X_test ,y_train , y_test