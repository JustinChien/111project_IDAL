import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense,Conv2D,MaxPooling2D,Dropout,Flatten,BatchNormalization,Input,LeakyReLU,Activation
from keras.applications.vgg19 import VGG19,preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.metrics import multilabel_confusion_matrix,classification_report,confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import config
import utils
import os

#USE GPU FOR TRAINGING
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("GPUs Avaliable{}:".format(physical_devices))
tf.config.experimental.set_memory_growth(physical_devices[0],True)



def create_model(units_1,
                 units_2,
                 dropout_rate_1,
                 dropout_rate_2,
                 use_GAP2D:bool,
                 model_name=config.model_name,
                 data_class=config.data_class,
                 IMG_SIZE=config.IMG_SIZE,
                 show_detail=config.show_detail):
    
    # Use the model name given as param to create the model. they will have the same img_aug, flattern and prediciton layer.
    # params with "*" means they will read from config.py if not given
    
    # * model name       : choose what model to create
    # * data_class       : control the number of prediction classes
    # * IMG_SIZE         : the img size that will be used for creating model
    # * show_detail      : decide whether to print out the summary of the model
    # units_1&2          : pass to the fully_connected_layer's Dense(units=)
    # dropout_rate_1&2   : pass to the fully_connected_layer's Dropout(rate=)
    # use GAP2D          : decide to use flatten or GlobalAveragePooling2D

    data_augmentation = tf.keras.Sequential([
        # tf.keras.layers.RandomFlip(mode="horizontal_and_vertical"), flip will fuck up the preprocessing we did
        tf.keras.layers.RandomZoom(height_factor=(0.2,-0.2),
                                    width_factor=(0.2,-0.2),
                                    fill_mode='nearest'),
        tf.keras.layers.RandomBrightness(factor=0.001)
    ],name="Data_augmentation")

    if model_name == "DenseNet":

        base_model = tf.keras.applications.DenseNet121(input_shape=IMG_SIZE,include_top=False,weights=None)
        base_model.load_weights("D:\\111project\\gitHub\\041-111project\\data\\models\\DenseNet-BC-121-32-no-top.h5")
        base_model.trainable = False

        preprocess_input = tf.keras.applications.densenet.preprocess_input

        flatten_layer = tf.keras.layers.Flatten()

        prediction_layer = tf.keras.layers.Dense(len(data_class),activation="softmax")

        inputs = tf.keras.Input(shape=IMG_SIZE)
        x = data_augmentation(inputs)
        x = preprocess_input(x)
        x = base_model(x)
        # Our FC layer
        flat1 = Flatten()(x)

        dense1 = Dense(units=units_1, use_bias=True)(flat1) #units=256
        batchnorm1 = BatchNormalization()(dense1)
        act1 = Activation(activation='relu')(batchnorm1)
        drop1 = Dropout(dropout_rate_1)(act1) #rate=0.5

        dense1 = Dense(units=units_2, use_bias=True)(flat1) #units=32
        batchnorm1 = BatchNormalization()(dense1)
        act1 = Activation(activation='relu')(batchnorm1)
        drop1 = Dropout(dropoout_rate_2)(act1) #rate=0.3
        # Output
        out = Dense(units=2, activation='softmax')(drop1)

        # Create Model
        model = Model(inputs=inputs, outputs=out)

        if show_detail == True:
            print("{}121 Model Created.".format(model_name))
            model.summary()

    else:     
        #--- --- --- --- 分隔線 --- --- --- ---#        

        if model_name == "MobileNet": #MobileNetV2

            #Rescale Pixel Values
            preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

            #MobileNet itself
            base_model = tf.keras.applications.MobileNetV2(input_shape = IMG_SIZE,
                                                            include_top=False,
                                                              weights="imagenet")
            base_model.trainable = False

        #--- --- --- --- 分隔線 --- --- --- ---#        

        elif model_name == "Inception": # InceptionV3

            #Rescale Pixel Values
            preprocess_input = tf.keras.applications.inception_v3.preprocess_input
            #Inception itself
            base_model=tf.keras.applications.inception_v3.InceptionV3(
                        include_top=False,
                        weights='imagenet',
                        input_tensor=None,
                        input_shape=IMG_SIZE,
                        pooling=None,
                        classes=1000,
                        classifier_activation='softmax'
            )

            base_model.trainable = False

        #--- --- --- --- 分隔線 --- --- --- ---#        

        elif model_name == "VGG": #VGG19

            #Rescale Pixel Values
            preprocess_input = tf.keras.applications.vgg19.preprocess_input
            #VGG19 itself
            base_model = VGG19(weights='imagenet',include_top=False)
            base_model.trainable = False   

        #--- --- --- --- 分隔線 --- --- --- ---#        

        if model_name == "ResNet": # ResNet152V2

            #Rescale Pixel Values
            preprocess_input = tf.keras.applications.resnet.preprocess_input

            #MobileNet itself
            base_model = tf.keras.applications.ResNet152V2(input_shape = IMG_SIZE,
                                                            include_top=False,
                                                              weights="imagenet")
            base_model.trainable = False

        #--- --- --- --- 分隔線 --- --- --- ---#        

            
        GAP2D_layer = tf.keras.layers.GlobalAveragePooling2D()
        flatten_layer = tf.keras.layers.Flatten()

        prediction_layer = tf.keras.Sequential([
            tf.keras.layers.Dense(units=units_1),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(tf.keras.activations.relu),
            tf.keras.layers.Dropout(dropout_rate_1),
            
            tf.keras.layers.Dense(units=units_2),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(tf.keras.activations.relu),
            tf.keras.layers.Dropout(dropout_rate_2),

            tf.keras.layers.Dense(len(data_class),activation="softmax")
        ],name="Prediction_layer")

        # # Assemble all together
        inputs = tf.keras.Input(shape=IMG_SIZE)
        x = data_augmentation(inputs)
        # x = preprocess_input(inputs)        
        x = preprocess_input(x)
        x = base_model(x,training=False)
        
        if use_GAP2D==True:
            x = GAP2D_layer(x)
        else:
            x = flatten_layer(x)
        
        outputs = prediction_layer(x)
        model = tf.keras.Model(inputs,outputs,name="Custom_{}".format(model_name))

        if show_detail == True:
            print("{} Model Created.".format(model_name))
            model.summary()
        
        return model
    
def create_tuner_model(hp):
    
    # This function will use keras tuner to create a set of hyperparameters and pass to create_model()
    # By doing so will make our old model tunable for keras tuner 
    # btw create_model() will not compile model, so we compile it here!

    units_1 = hp.Int("units_1",max_value=512,min_value=16,step=16)
    units_2 = hp.Int("units_2",max_value=512,min_value=16,step=16)
    dropout_rate_1 = hp.Float("dropout_rate_1",max_value=0.9,min_value=0)
    dropout_rate_2 = hp.Float("dropout_rate_2",max_value=0.9,min_value=0)
    learning_rate = hp.Float("Learning_Rate",max_value=0.1,min_value=1e-08,sampling="log")
    use_GAP2D = hp.Boolean("use_GAP2D")

    model = create_model(units_1=units_1,
                         units_2=units_2,
                         dropout_rate_1=dropout_rate_1,
                         dropout_rate_2=dropout_rate_2,
                         use_GAP2D=use_GAP2D)
    
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate= learning_rate),
                  loss = tf.keras.losses.categorical_crossentropy, 
                  metrics=["accuracy"])

    return model
    
def run_through(model, X_train, y_train, X_test, y_test, lr, epochs, BATCH_SIZE):
    
    # run_through will take the model to compile,evaluate,fit,predict and return the history of the trainging and the prediction. 
    
    # model      : the model you want to train
    # X_train    : imgs of the training dataset
    # y_train    : loabel of the training dataset
    # X_test     : imgs of the test dataset
    # y_test     : label of the test dataset
    # lr         : learning rate for optimizer
    # epcohs     : epoch for compile model
    # BATCH_SIZE : BATCH_SIZE for compile model
    
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate= lr),loss = tf.keras.losses.binary_crossentropy, metrics=["accuracy"])
    
    loss0,accuracy0 = model.evaluate(x=X_test,y=y_test,batch_size=BATCH_SIZE)

    print("initial loss: {:.2f}".format(loss0))
    print("initial accuracy: {:.2f}".format(accuracy0))
    
    history = model.fit(x=X_train, y=y_train, epochs=epochs, validation_data=(X_test,y_test), batch_size=BATCH_SIZE, shuffle=True)
    
    predictions = model.predict(x=X_test)
    
    return history,predictions

def demo_img_aug():
    
    # Demonstration of what our Image Augmentation will do to the data and how it looks like after augmented.
    # Use the Okay. image to better understand / see what part of the picture has been changed.
    
    da_test = tf.keras.Sequential([
        # tf.keras.layers.RandomFlip(mode="horizontal_and_vertical"),
        tf.keras.layers.RandomZoom(height_factor=(0.2,-0.2),
                                    width_factor=(0.2,-0.2),
                                    fill_mode='nearest'),
        tf.keras.layers.RandomBrightness(factor=0.001)
    ],name="Data_augmentation")
    
    
    #this show what we've done in image augmentation
    img = image.load_img("D:\\111project\\github\\041-111project\\data\\test.png",target_size=(224,224,3))
    img = image.img_to_array(img)

    img = img/255.
    plt.figure(figsize=(20,20))
    for i in range(25):
        aug_img = da_test(img,training=True)

        ax = plt.subplot(5,5,i+1)
        plt.imshow(aug_img)
        plt.axis("off")
    return 

def show_performance(history, predictions, X_train, y_train, X_test, y_test, note="", classes=config.data_class, show_detail=config.show_detail, save_in_csv=False):
    
    # Create tables,plots to show how good does it perform with the dataset
    # save_in_csv will decide whether to save the performance report in report_csv
    # params with "*" means they will read from config.py if not given
    
    # history       : the history object you'll get from training model
    # predicitons   : the prediction of your model which is from model.predict()
    # X_train       : imgs of the training dataset
    # y_train       : loabel of the training dataset
    # X_test        : imgs of the test dataset
    # y_test        : label of the test dataset
    # note          : a string that will add to the report. will be empyty str if not given
    # * classes     : classes of the dataset you use.
    # * show_detail : whether to print out the detail or not
    # * save_in_csv : whether to save in the report csv
    
    y_pred=np.argmax(predictions,axis=-1)
    y_true=np.argmax(y_test,axis=-1)
    cm = confusion_matrix(y_true=y_true,y_pred=y_pred)
    cm_container = utils.convert_cm_to_container(cm)
    report = classification_report(y_true=y_true,y_pred=y_pred,target_names=config.data_class)
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    if show_detail:
        print(report)
        utils.plot_acc_loss(acc,val_acc,loss,val_loss)
        utils.plot_confusion_matrix(cm,classes)
        
    if save_in_csv:
        history_report = pd.read_csv(config.report_csv_path)
        history_report.loc[len(history_report)]=(time.ctime(time.time()),
                                                    config.model_name,
                                                    config.batch_size,
                                                    config.epochs,
                                                    config.learning_rate,
                                                    report,
                                                    cm_container,
                                                    utils.convert_list_to_str(acc),
                                                    utils.convert_list_to_str(val_acc),
                                                    utils.convert_list_to_str(loss),
                                                    utils.convert_list_to_str(val_loss),
                                                    config.data_type,
                                                    note)
        history_report.to_csv(config.report_csv_path,index=False)
    return 
        
def load_model(show_detail = config.show_detail):
    
    # load models in data\models file folder
    # will list out options and input an int to choose.
    # params with "*" means they will read from config.py if not given

    # * show_detail : whether to print out the detail or not

    models_path = "D:\\111project\\github\\041-111project\\data\\models"
    current_opt = os.listdir(models_path)
    for i in range(len(current_opt)):
        print("{}) : {}".format(i,current_opt[i]))
    try:    
        model_ = keras.models.load_model(utils.join_path(models_path,current_opt[int(input("Pls choose the model you want:"))]))
    except:
        print("This file is not exist \ usable.")
        
    model_.build(input_shape=config.IMG_SIZE)
    
    if show_detail:
        model_.summary()
    
    return model_

def evaluate_model(model,X_train, X_test ,y_train , y_test):
    
    # input a model and train\test data
    # will start a simplified evaluate 
    # output with a Classfication report and ConfusionMatrix plot
    
    # model      : the model you want to train
    # X_train    : imgs of the training dataset
    # y_train    : loabel of the training dataset
    # X_test     : imgs of the test dataset
    # y_test     : label of the test dataset
    
    loss0,accuracy0 = model.evaluate(x=X_test,y=y_test,batch_size=config.batch_size)
    predictions = model.predict(x=X_test)
    
    y_pred=np.argmax(predictions,axis=-1)
    y_true=np.argmax(y_test,axis=-1)
    cm = confusion_matrix(y_true=y_true,y_pred=y_pred)
    cm_container = utils.convert_cm_to_container(cm)
    report = classification_report(y_true=y_true,y_pred=y_pred,target_names=config.data_class)
    
    print(report)
    utils.plot_confusion_matrix(cm,classes=config.data_class)