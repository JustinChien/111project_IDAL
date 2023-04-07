import io
import os
import cv2
import base64
import requests
import face_recognition
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# This python file will contain all the functions for usage in flask-app.py
# including using OpenCV to compare two imgs, ...
# make sure to fulfill the format of each paremeters.

# All request to API use this URL

known_faces = []
know_name = []
recognition_threshold = 0.4
Face_Library_Path='./Face_Library/'
url = 'http://120.101.3.229:3000/data'

def decode_base64(input_img, if_resize=True, img_size=(224,224)):
    # parameters:
    # input_img : an image in base64 stream
    # if_resize : whether to resize or not
    # img_size : the target size for resize, if not given will have (224,224)

    # Decode the base64 encoded image data into bytes
    image_bytes = base64.b64decode(input_img)
    image = Image.open(io.BytesIO(image_bytes))
    if if_resize:
        image = image.resize(img_size)
    input_data = np.array(image)
    input_data = np.expand_dims(input_data, axis=0)
    return input_data

def encode_base64(input_img):
    # receive an image and encode it into base64 then return.
    _, buffer = cv2.imencode('.jpg', input_img)
    base64_img = base64.b64encode(buffer).decode('utf-8')
    return base64_img

def get_data(queryType:str,IDs='') -> list:

    # param will be the queryType to choose and please give IDs if its byRID or byPID
    # queryType : patient,byRID,byPID
    # IDs : <num>     for queryType = byPID
    #       <num,num> for queryType = byRID
    # DONT TYPE THOSE BRACKETS.

    #choose what data to get ccording to input
    if queryType == 'patient':
        params = {  'queryType':'patient'}
    elif queryType == 'byPID':
        params = {  'queryType':'byPID',
                    'patientID':str(IDs)}
    elif queryType == 'byRID':
        params = {  'queryType': 'byRID',
                    'reportID': str(IDs)}

    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()
        return data
    else:
        print('Request failed with status code:', response.status_code)

def compare_img(path1:str,path2:str):

    # Load two path's image and return a compare result with marking
    # path1 : the first image's path
    # path2 : the second image's path

    # !!! PLEASE MAKE SURE THIS TWO IMAGE ARE FROM SAME PATIENT. !!!

    # 读取两张图片
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)

    # 将两张图片转换为灰度图
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 计算两张图片的差异
    diff = cv2.absdiff(gray1, gray2)

    # 对差异图像进行二值化处理
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    # 对二值化图像进行膨胀操作
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    thresh = cv2.dilate(thresh, kernel)

    # 找到差异区域的轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 在原图上绘制差异区域的矩形框
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(img1, (x, y), (x+w, y+h), (0, 0, 255), 2)

    # 显示结果
    # # resize the image first, cause it's kinda big.
    # resize_img = cv2.resize(img1,(800,800))
    # cv2.imshow('Compare_Result',resize_img)
    # cv2.waitKey(0)
    return img1

def generate_cropped_img(img_path:str,roi_path:str) -> list:

    # Load the image and its correspond ROI image
    # return a list of cropped image
    # img_path : path of the X-ray image
    # roi_path ; path of the X-ray image's ROI image

    result = list()
    # 讀取原圖和ROI圖
    img = cv2.imread(img_path)
    roi_img = cv2.imread(roi_path,0 ) # flag 0 will read as grey-scale image

    # 转换ROI图为二值化图像
    ret, roi_img = cv2.threshold(roi_img, 127, 255, cv2.THRESH_BINARY)

    # 获取ROI图中的所有轮廓
    contours, hierarchy = cv2.findContours(roi_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 循环每个轮廓并裁剪原始图像
    for contour in contours:
        # 获取轮廓的最小边界框坐标和宽度/高度
        x, y, w, h = cv2.boundingRect(contour)
        # 使用坐标和宽度/高度裁剪原始图像
        roi = img[y:y+h, x:x+w]
        result.append(roi)
    return result

def marking_abnormal(img_path:str,roi_path:str):

    # Marking the abnormal part of the X-ray and displaying some infomation about it.
    # Return an image with all the markings and info.
    # img_path : path of the X-ray image
    # roi_path : path of the X-ray image's ROI image
    img = cv2.imread(img_path)
    roi_img = cv2.imread(roi_path,0)

    # resize the image into readable size, too big will result in small text and can't show the whole
    # image on the screen.
    img = cv2.resize(img,(2000,2000))
    roi_img = cv2.resize(roi_img,(2000,2000))

    c = []

    _, roi_img = cv2.threshold(roi_img, 128, 255, cv2.THRESH_BINARY)

    # 找到所有轮廓
    contours, hierarchy = cv2.findContours(roi_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_contours = len(contours)

    # 框出每个轮廓
    for cnt in contours:
        #取得繞著輪廓的多邊形頂點
        poly = cv2.approxPolyDP(cnt, 3, True)

        # 繞著ROI圖繪製線條
        for i in range(len(poly)):
            cv2.line(img, tuple(poly[i][0]), tuple(poly[(i+1)%len(poly)][0]), (0, 255, 0), 2)
        # 获取轮廓在ROI图像中的坐标
        x, y, w, h = cv2.boundingRect(cnt)

        # 在原图中框出每个轮廓
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # 裁切出每个轮廓在原图中对应的位置
        roi_contour = img[y:y+h, x:x+w]
        c.append(roi_contour)

        perimeter = cv2.arcLength(cnt, True)

        # 计算近似多边形并计算面积
        epsilon = 0.01 * perimeter
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        area = cv2.contourArea(approx)

        # 計算輪廓長度、寬度、面積、mean、min等資訊
        # length, width = calculate_contour_length_width(contour)
        x, y, w, h = cv2.boundingRect(cnt)

        # 计算矩形长度和宽度
        length = max(w, h)
        width = min(w, h)

        #計算密度
        density = np.sum(roi_img == 255) / (w * h)

        # area = cv2.contourArea(contour)
        mean_val = cv2.mean(img, mask=roi_img)[0]
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(roi_img, mask=None)

        # 在原圖上標註輪廓外接矩形和文字
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(img, f"Length: {length:.2f}", (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(img, f"Width: {width:.2f}", (x, y - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(img, f"Area: {area:.2f}", (x, y - 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(img, f"Mean: {mean_val:.2f}", (x, y - 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(img, f"Min: {min_val:.2f}", (x, y - 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(img, f"Perimeter: {perimeter:.2f}", (x, y - 170), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(img, f"Density: {density:.2f}", (x, y - 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return img

def plot_imgs(imgs):

    # will plot the first 9 imgs

    # imgs : list of image to plot

    plt.figure(figsize=(15,15))
    for i in range(len(imgs)):
        ax = plt.subplot(len(imgs),1,i+1)
        plt.imshow(imgs[i])
        plt.axis("off")

def facenet_init(if_show=False) -> None:

    # this funciton will initialize the Facenet Recognition service
    # create a face library and use the image's filename as that face's name

    # parameters:
    # if_show : boolean, whether to print out the face's name and number of face library

    for i in os.listdir(Face_Library_Path):
        face=face_recognition.load_image_file(Face_Library_Path+i)
        f=face_recognition.face_encodings(face)[0]
        known_faces.append(f)
        know_name.append(i[0:len(i)-4])
    if if_show:
        print("Detected Faces:{}".format(len(known_faces)))
        print(know_name)
    return

def facenet_recognize(image):

    # 還沒寫例外，辨識不出人臉的例外

    # given an image of face and it will recognize the face.
    # If can't recognize as someone in face library return None.

    # parameters:
    # image : a 3D ndarray of a face image.

    unknown_face_encoding = face_recognition.face_encodings(image)[0]
    results = face_recognition.compare_faces(known_faces, unknown_face_encoding,tolerance=recognition_threshold)
    ans_dict=dict(zip(know_name,results)) # combine the result of recognition and the face library
    for ppl in ans_dict: # ppl is a str of someone's name
        if ans_dict[ppl] == True:
            return ppl

# initialize the FaceNet Service
facenet_init()