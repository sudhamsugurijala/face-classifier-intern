# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 14:28:23 2019

@author: Internship007
"""
# importing necessary packages
from os import listdir
import pickle
import json
import sys
from datetime import datetime
from numpy import load
from numpy import expand_dims
from numpy import asarray
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from cv2 import cv2
from PIL import Image, ImageEnhance
from mtcnn.mtcnn import MTCNN
from keras.models import load_model
import filetype


def convert(lst):
    """Convert list into dictionary."""
    res_dct = {lst[i]: lst[i + 1] for i in range(0, len(lst), 2)}
    return res_dct


def get_embedding(model, face_pixels):
    """Function to get embeddings of a face."""
    # scale pixel values
    face_pixels = face_pixels.astype('float32')
    # standardize pixel values across channels (global)
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    # transform face into one sample
    samples = expand_dims(face_pixels, axis=0)
    # make prediction to get embedding
    yhat = model.predict(samples)
    return yhat[0]


def pred_face(face_array, dict_faces):
    """Function to predict face of a person."""
    # set default threshold as 90%
    if len(sys.argv) < 3:
        threshold = 90
    else:
        threshold = int(sys.argv[2])
    dict_face = {}
    # to store face embedding of test_image
    test_sample_emb = []
    # load model to get embeddings
    model = load_model('facenet_keras.h5')
    # to get embeddings of test_image
    embedding = get_embedding(model, face_array)
    #  convert test_image embeddings into an array
    test_sample_emb = asarray(embedding)
    # convert test_image embeddings into 2-D array
    test_sample_emb1 = [test_sample_emb]
    # open the file of trained model
    file = open('svm_model.h5', 'rb')
    # load the file of trained model
    model = pickle.load(file)
    file.close()
    # load face embeddings of trained data
    data = load('faces-embeddings.npz')
    trainx, trainy = data['arr_0'], data['arr_1']
    # normalize input vectors
    in_encoder = Normalizer(norm='l2')
    trainx = in_encoder.transform(trainx)
    test_sample_emb1 = in_encoder.transform(test_sample_emb1)
    # label encode targets
    out_encoder = LabelEncoder()
    out_encoder.fit(trainy)
    trainy = out_encoder.transform(trainy)
    # for finding unique labels
    copy = []
    for x_1 in trainy:
        if x_1 not in copy:
            copy.append(x_1)
    # convert labels into corresponding names of person
    copy = out_encoder.inverse_transform(copy)
    sample = expand_dims(test_sample_emb1[0], axis=0)
    # predict label of face
    y_class = model.predict(sample)
    # predict probability of face
    y_prob = model.predict_proba(sample)
    # print face classification probability
    print(y_prob)
    class_index = y_class[0]
    # to print main probability of predicted sample
    class_probability = y_prob[0, class_index] * 100
    # to convert labels into corresponding names of predicted persons
    predict_names = out_encoder.inverse_transform(y_class)
    print(predict_names[0])
    if class_probability > threshold:
        print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))
        name = predict_names[0]
        pred_name = predict_names[0]
        pred_prob = class_probability
        face = pred_name+' '+'(%.3f)' % pred_prob
        mainlist = []
        # split probability between all classes in mainlist
        for i, j in enumerate(copy):
            mainlist.append(j)
            mainlist.append(y_prob[0][i] * 100)
        # getting system date and time
        now = datetime.now()
        date_time = now.strftime("%d/%m/%Y, %H:%M:%S")
        face = face+'  '+date_time
        dict_list = convert(mainlist)
        dict_face.update({face: dict_list})
        dict_faces.update(dict_face)
        return name
    print('Predicted: Unknown')
    return 'Unknown'


def predict_sample(path, dict_faces, filename):
    """Predict sample."""
    # open image
    image = Image.open(path)
    # enhance brightness of image
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(1.8)
    # conversion of images into pixels
    pixels = asarray(image)
    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    results = detector.detect_faces(pixels)
    # open image using openCV
    image_out = cv2.imread(path)
    # extract the bounding box from every face
    # i variable not used, so '_' is used
    for _, j in enumerate(results):
        x_1, y_1, w_1, h_1 = j['box']
        # bug fix
        x_1, y_1 = abs(x_1), abs(y_1)
        # neglect small faces
        if w_1 < 90 or h_1 < 120:
            continue
        x_2, y_2 = x_1 + w_1, y_1 + h_1
        # extract the face
        face = pixels[y_1:y_2, x_1:x_2]
        # resize pixels to the model size
        image = Image.fromarray(face)
        image = image.resize((160, 160))
        face_array = asarray(image)
        name = pred_face(face_array, dict_faces)
        # to draw a box around the detected face
        image_out = cv2.rectangle(image_out, (x_1, y_1),
                                  (x_1 + w_1, y_1 + h_1),
                                  (36, 255, 12), 1)
        cv2.putText(image_out, name, (x_1, y_1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (36, 255, 12), 2)
        # to display the output
    cv2.imshow('', image_out)
    # to write output images into new folder
    cv2.imwrite('O1/ % s' % filename, image_out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def predict_data(directory, dict_out):
    """Predict data."""
    if len(listdir(directory)) == 0:
        return -1
    for filename in listdir(directory):
        dict_faces = {}
        # path
        path = directory + filename
        # filetype.guess NEEDS ABSOLUTE PATH AS PARAMETER
        file_type = filetype.guess(path)
        if file_type.mime.find('image') > -1:
            # passing 0 for image
            predict_sample(path, dict_faces, filename)
        else:
            continue
        dict_out.update({filename: dict_faces})
    return 0


# first cmd line argument is path to testing data
DIR_PATH = '%s' % sys.argv[1] + '/'
# dictionary to write into JSON file
DICT_OUT = {}
NUM_FILES = predict_data(DIR_PATH, DICT_OUT)
if NUM_FILES == -1:
    print('The Directory is EMPTY!!')
    # write into JSON file
with open('test_out1.json', 'w') as file1:
    # indent = 2 for newline in JSON file
    json.dump(DICT_OUT, file1, indent=2)
