import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense,Conv2D,MaxPooling2D,Dropout,Flatten,BatchNormalization,Input,LeakyReLU,Activation
from keras.applications.vgg19 import VGG19,preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.metrics import multilabel_confusion_matrix,classification_report,confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

import config
import utils

#USE GPU FOR TRAINGING
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("GPUs Avaliable{}:".format(physical_devices))
tf.config.experimental.set_memory_growth(physical_devices[0],True)


def create_model(model_name ,data_class ,IMG_SIZE ,show_detail=False):
    # Use the model name given as param to create the model. they will have the same img_aug, flattern and prediciton layer.
    # model name will choose what model to create
    # data_class will control the number of prediction classes
    # show_detail will decide whether to print out the summary of the model

    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip(mode="horizontal_and_vertical"),
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

        dense1 = Dense(units=256, use_bias=True)(flat1)
        batchnorm1 = BatchNormalization()(dense1)
        act1 = Activation(activation='relu')(batchnorm1)
        drop1 = Dropout(rate=0.5)(act1)

        dense1 = Dense(units=32, use_bias=True)(flat1)
        batchnorm1 = BatchNormalization()(dense1)
        act1 = Activation(activation='relu')(batchnorm1)
        drop1 = Dropout(rate=0.3)(act1)
        # Output
        out = Dense(units=2, activation='softmax')(drop1)

        # Create Model
        model = Model(inputs=inputs, outputs=out)

        if show_detail == True:
            print("{}121 Model Created.".format(model_name))
            model.summary()

    else:     
        #--- --- --- --- 分隔線 --- --- --- ---#        

        if model_name == "MobileNet":

            #Rescale Pixel Values
            preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

            #MobileNet itself
            base_model = tf.keras.applications.MobileNetV2(input_shape = IMG_SIZE,
                                                            include_top=False,
                                                              weights="imagenet")
            base_model.trainable = False

        #--- --- --- --- 分隔線 --- --- --- ---#        

        elif model_name == "Inception":

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

        elif model_name == "VGG":

            #Rescale Pixel Values
            preprocess_input = tf.keras.applications.vgg19.preprocess_input
            #VGG19 itself
            base_model = VGG19(weights='imagenet',include_top=False)
            base_model.trainable = False   

        #--- --- --- --- 分隔線 --- --- --- ---#        

        # global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        flatten_layer = tf.keras.layers.Flatten()

        prediction_layer = tf.keras.Sequential([
            tf.keras.layers.Dense(256,activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(128,activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            # x = Dense(64,activation='relu'),
            # x = BatchNormalization(),
            # x = Dropout(0.2),
            tf.keras.layers.Dense(len(data_class),activation="softmax"),
        ],name="Prediction_layer")

        # # Assemble all together
        inputs = tf.keras.Input(shape=IMG_SIZE)
        x = data_augmentation(inputs)
        x = preprocess_input(x)
        x = base_model(x,training=False)
        x = flatten_layer(x)
        outputs = prediction_layer(x)
        model = tf.keras.Model(inputs,outputs)

        if show_detail == True:
            print("{} Model Created.".format(model_name))
            model.summary()
            
        return model
    
def run_through(model, X_train, y_train, X_test, y_test, lr, epochs, BATCH_SIZE):
    # run_through will take the model to compile,evaluate,fit,predict and return the history of the trainging and the prediction. 
    # model is the one you want to train
    # X_train, y_train as the training dataset
    # X_test, y_test as the validation dataset
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
        tf.keras.layers.RandomFlip(mode="horizontal_and_vertical"),
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
        
def show_performance(history, predictions, X_train, y_train, X_test, y_test, show_detail=False, save_in_csv=False, note=""):
    
    # Create tables,plots to show how good does it perform with the dataset
    # save_in_csv will decide whether to save the performance report in report_csv
    # note is the Note that will added with the report. Write sth down to describe \ explain the record.
    
    y_pred=np.argmax(predictions,axis=-1)
    y_true=np.argmax(y_test,axis=-1)
    cm = confusion_matrix(y_true=y_true,y_pred=y_pred)
    cm_container = utils.convert_cm_to_container(cm)
    report = classification_report(y_true=y_true,y_pred=y_pred,target_names=DATA_CLASS)
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    if show_detail:
        print(report)
        utils.plot_acc_loss(acc,val_acc,loss,val_loss)
        utils.plot_confusion_matrix(cm)
        
    if save_in_csv:
    history_report = pd.read_csv(config.report_csv_path)
    history_report.loc[len(history_report)]=(time.ctime(time.time()),
                                                model_name,
                                                BATCH_SIZE,
                                                epochs,
                                                learning_rate,
                                                report,
                                                cm_container,
                                                utils.convert_list_to_str(acc),
                                                utils.convert_list_to_str(val_acc),
                                                utils.convert_list_to_str(loss),
                                                utils.convert_list_to_str(val_loss),
                                                DATA_TYPE,
                                                note)
    history_report.to_csv(report_csv_path,index=False)