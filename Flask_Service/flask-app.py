import tensorflow as tf
import numpy as np
from flask import Flask, request, jsonify
import function as func

input_shape = (224, 224)
host_ip = '120.101.3.229'

# Load the saved TensorFlow model
calc_model = tf.keras.models.load_model('D:/111project/github/041-111project/Models/calc_pre_G_4_efficientnetB7model.h5')
# mass_model = tf.keras.models.load_model('D:/111project/github/041-111project/Models/mass_pre_G_4_efficientnetB7model_22.h5')

# Create the Flask app
app = Flask(__name__)

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # parameters:
    # input_data : an image that encoded in base64
    #              should be someone's mammography image.
    # data_type : str, to choose use which model to predict

    input_data = request.json['data']
    # data_type = request.json['type']

    # Preprocess the input data if needed
    preprocessed_data = func.decode_base64(input_data,if_resize=True,img_size=(224,224))

    predictions = calc_model.predict(preprocessed_data)
    # 等老師討論後決定要架 1 or 2個模型
    # if data_type == 'calc' or 'Calc':
    #     predictions = calc_model.predict(preprocessed_data)
    # elif data_type == 'mass' or 'Mass':
    #     predictions = mass_model.predict(preprocessed_data)

    # Get the predict value and translate it into Answer,like 'Cat' and 'Dog'
    processed_output = np.argmax(predictions,axis=-1)
    if processed_output == 1:
        return jsonify({'predictions': "Malignant"})
    else:
        return jsonify({'predictions': "Benign"})

@app.route('/compare/compare_image', methods=['POST'])
def compare():
    # parameters:
    # reportID_1 : <int> // the reportID from the same patient
    # reportID_2 : <int> // but DIFFERENT date.
    # img_type :  CC,MLO // to specify what type of image to compare,
    reportID_1 = request.json['id1']
    reportID_2 = request.json['id2']
    img_type = request.json['img_type']

    data = func.get_data(queryType='byRID',IDs='{},{}'.format(reportID_1,reportID_2))
    compare_result = func.compare_img(data[0]['mammo_{}_imgPath'.format(img_type)],data[1]['mammo_{}_imgPath'.format(img_type)])

    # encode image into base64 and return
    compare_result_base64 = func.encode_base64(compare_result)
    return jsonify({'image': compare_result_base64})

@app.route('/compare/generate_cropped', methods=['POST'])
def generate_cropped():
    # parameters:
    # reportID : <int> // the reportID of a patient's report
    # img_type :  CC,MLO // to specify what type of image to compare,

    reportID = request.json['id']
    img_type = request.json['img_type'] # to specify whether to use CC or MLO image

    data = func.get_data(queryType='byRID',IDs='{}'.format(reportID))
    cropped_images = func.generate_cropped_img(data[0]['mammo_{}_imgPath'.format(img_type)],data[0]['mammo_{}_ROI_imgPath'.format(img_type)])

     # If the result is a single image, return it as a base64-encoded string
    if len(cropped_images) == 1:
        cropped_image = cropped_images[0]
        cropped_image_base64 = func.encode_base64(cropped_image)
        return jsonify({'image': cropped_image_base64})

    # If the result is a list of images, return them as a list of base64-encoded strings
    elif len(cropped_images) > 1:
        cropped_images_base64 = []
        for cropped_image in cropped_images:
            base64_image = func.encode_base64(cropped_image)
            cropped_images_base64.append(base64_image)
        return jsonify({'image': cropped_images_base64})

@app.route('/compare/marking_abnormal', methods=['POST'])
def marking_abnormal():
    # parameters:
    # reportID : <int> // the reportID of a patient's report
    # img_type :  CC,MLO // to specify what type of image to compare,

    reportID = request.json['id']
    img_type = request.json['img_type'] # to specify whether to use CC or MLO image

    data = func.get_data(queryType='byRID',IDs=reportID)
    marked_img = func.marking_abnormal(data[0]['mammo_{}_imgPath'.format(img_type)],data[0]['mammo_{}_ROI_imgPath'.format(img_type)])
    marked_img_base64 = func.encode_base64(marked_img)
    return jsonify({'image':marked_img_base64})

@app.route('/facenet', methods=['POST'])
def facenet():
    # parameters:
    # image : the face image used to recognize
    # this function will return the patientID of the people in the image
    # if it's not recognized, return none
    base64_image = request.json['image']
    image = func.decode_base64(base64_image,if_resize=False)[0]
    patientID = func.facenet_recognize(image)
    return jsonify({'PID': patientID})

# Run the app
if __name__ == '__main__':
    # app.debug = True
    app.run(host=host_ip,port=5000)
