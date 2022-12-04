#Constants
report_csv_path = "D:\\111project\\gitHub\\041-111project\\data\\history_report_CalcOnly.csv"
IMG_DIR = {"ALL":"D:\\111project\\gitHub\\041-111project\\data\\Cropped_Calc_imgs",
           "CC":"D:\\111project\\gitHub\\041-111project\\data\\Cropped_Calc_imgs",
           "MLO":"D:\\111project\\gitHub\\041-111project\\data\\Cropped_Calc_imgs",
           "FULL_ALL":"D:\\111project\\gitHub\\041-111project\\data\\fullimage_preprocessing",
           "FULL_CC":"D:\\111project\\gitHub\\041-111project\\data\\fullimage_preprocessing",
           "FULL_MLO":"D:\\111project\\gitHub\\041-111project\\data\\fullimage_preprocessing",
           "CatnDog":"D:\\111project\\gitHub\\041-111project\\data\\CatsandDogs\\train"}

data_csv_path = {"ALL":"D:\\111project\\gitHub\\041-111project\\data\\Calc_ALL.csv",
                 "CC":"D:\\111project\\gitHub\\041-111project\\data\\Calc_CC.csv",
                 "MLO":"D:\\111project\\gitHub\\041-111project\\data\\Calc_MLO.csv",
                 "FULL_ALL":"D:\\111project\\gitHub\\041-111project\\data\\Full_Calc_ALL.csv",
                 "FULL_CC":"D:\\111project\\gitHub\\041-111project\\data\\Full_Calc_CC.csv",
                 "FULL_MLO":"D:\\111project\\gitHub\\041-111project\\data\\Full_Calc_MLO.csv",
                 "CatnDog":"D:\\111project\\gitHub\\041-111project\\data\\CatsandDogs\\train.csv"}
IMG_SIZE = (224,224,3)
R_SEED = 12

#Variables
data_type = "FULL_ALL" #options: ALL,CC,MLO,FULL_ALL,FULL_CC,FULL_MLO,CatnDog
data_class = ['Benign','Malignant']
model_name = "ResNet" #Options: MobileNet,Inception,VGG,DenseNet,ResNet
learning_rate = 1e-05
epochs = 30
batch_size = 32
show_detail = True