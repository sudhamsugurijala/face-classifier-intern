# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 15:29:01 2019

@author: Internship007
"""
# importing necessary packages
from os import listdir, mkdir, getcwd
from os.path import isdir, exists
import sys
import pickle
import shutil  # for removing dir with contents
from PIL import Image, ImageEnhance
from numpy import savez_compressed
from numpy import asarray
from numpy import load
from numpy import expand_dims
from mtcnn.mtcnn import MTCNN
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator, img_to_array
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC


def extract_face(filename, required_size=(160, 160)):
    """Extract a single face from a given photograph."""
    # load image from FILE
    image = Image.open(filename)
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(1.8)
    pixels = asarray(image)
    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    results = detector.detect_faces(pixels)
    # extract the bounding box from the first face
    # bug fix for list index out of range
    try:
        x_1, y_1, width, height = results[0]['box']
    except IndexError:
        return None
    x_1, y_1 = abs(x_1), abs(y_1)
    x_2, y_2 = x_1 + width, y_1 + height
    # extract the face
    face = pixels[y_1:y_2, x_1:x_2]
    # resize pixels to the MODEL size
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = asarray(image)
    return face_array


def load_faces(directory):
    """Load images and extract faces for all images in a directory."""
    faces = list()
    # enumerate FILEs
    for filename in listdir(directory):
        # path
        path = directory + filename
        # data generator for augmentation
        datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1,
                                     height_shift_range=0.1, shear_range=0.15,
                                     zoom_range=0.1, channel_shift_range=10,
                                     horizontal_flip=True)
        image = Image.open(path)
        image = img_to_array(image)
        image = expand_dims(image, 0)
        # create new directory for augmentations in current working dir(cwd)
        dest = getcwd() + '/aug'
        mkdir(dest)
        datagen.fit(image)
        # to set default value of number of augmentations if not specified
        if len(sys.argv) < 3:
            augs_num = 30
        else:
            augs_num = int(sys.argv[2])
        i = 0
        # loop variable for augmentation
        for _ in datagen.flow(image, batch_size=1, save_to_dir=dest,
                              save_prefix='aug', save_format='jpg'):
            i += 1
            # second argument in cmd line will be number of iterations
            if i == augs_num:
                break
        dest = dest + '/'
        for augfile in listdir(dest):
            path = dest + augfile
            face = extract_face(path)
            # store
            faces.append(face)
        # remove directory after face extraction
        shutil.rmtree(dest)
    return faces


def load_dataset(directory):
    """Load DATAset containing one subdir for each person"""
    x_1, y_1 = list(), list()
    # enumerate folders, onw per class
    for subdir in listdir(directory):
        # path
        path = directory + subdir + '/'
        # skip any FILEs that might be in the dir
        if not isdir(path):
            continue
        # load all not None faces
        faces = load_faces(path)
        new_faces = list()
        for face in faces:
            if face is not None:
                new_faces.append(face)
        # create labels
        labels = [subdir for _ in range(len(new_faces))]
        # summarize progress
        print('loaded %d examples for class: %s' % (len(new_faces), subdir))
        # store
        x_1.extend(new_faces)
        y_1.extend(labels)
    return asarray(x_1), asarray(y_1)


def get_embedding(model, face_pixels):
    """Get the face embedding for one face"""
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


# deleting aug folder if it exists
if exists('%s/aug' % getcwd()):
    shutil.rmtree('%s/aug' % getcwd())
# load train DATAset
# first argument of cmd line is path to training DATA
TRAINX, TRAINY = load_dataset('%s' % sys.argv[1]+'/')

print(TRAINX.shape, TRAINY.shape)

# save arrays to one FILE in compressed format
savez_compressed('DATAset.npz', TRAINX, TRAINY)
DATA = load('DATAset.npz')

TRAINX, TRAINY = DATA['arr_0'], DATA['arr_1']

print('Loaded: ', TRAINX.shape, TRAINY.shape)
# load the facenet MODEL
MODEL = load_model('facenet_keras.h5')
print('Loaded MODEL')
# convert each face in the train set to an embedding
NEWTRAINX = list()
for face_pixls in TRAINX:
    embedding = get_embedding(MODEL, face_pixls)
    NEWTRAINX.append(embedding)
NEWTRAINX = asarray(NEWTRAINX)
print(NEWTRAINX.shape)

# save arrays to one FILE in compressed format
savez_compressed('faces-embeddings.npz', NEWTRAINX, TRAINY)

# Load dataset
DATA = load('faces-embeddings.npz')
TRAINX, TRAINY = DATA['arr_0'], DATA['arr_1']
print('Dataset: train=%d' % (TRAINX.shape[0]))

# normalize input vectors
IN_ENCODER = Normalizer(norm='l2')
TRAINX = IN_ENCODER.transform(TRAINX)
# label encode targets
OUT_ENCODER = LabelEncoder()
OUT_ENCODER.fit(TRAINY)
TRAINY = OUT_ENCODER.transform(TRAINY)

# fit MODEL
MODEL = SVC(kernel='linear', probability=True)
MODEL.fit(TRAINX, TRAINY)

# save MODEL,pickle it
FILE = open('svm_model.h5', 'wb')
pickle.dump(MODEL, FILE)
FILE.close()

# predict
PREDICT_TRAIN = MODEL.predict(TRAINX)

# score
SCORE_TRAIN = accuracy_score(TRAINY, PREDICT_TRAIN)

# summarize
print('Accuracy: train=%.3f' % (SCORE_TRAIN*100))
