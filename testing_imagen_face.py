
import pickle

import os
import configparser
import torch
import numpy as np
from transformers import AutoProcessor, ASTModel
from scipy.io import wavfile
import torch
import pickle
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import ShortTermFeatures
import librosa
import subprocess
import os
import pathlib
import configparser
import cv2
import random
from deepface import DeepFace
import matplotlib.pyplot as plt
import sqlite3 as sl
from PIL import Image
import numpy as np
import math
import threading
import time
from multiprocessing import Process
from threading import Thread




def resizeImage(im, size):
    
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(im.astype('uint8'), 'RGB')
    x, y = image.size
    if(x >= y):
      x_n = size
      y_n = int(math.floor((x_n/x) * y))
    else:
      y_n = size
      x_n = int(math.floor((y_n/y) * x))
    image_n = image.resize((x_n,y_n), Image.ANTIALIAS)       
    return image_n

def make_square(im, min_size, fill_color=(0, 0, 0, 0)):
    x, y = im.size
    size = max(min_size, x, y)
    new_im = Image.new('RGBA', (size, size), fill_color)
    new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
    return new_im

def extract_face(q,absPathVideo,resizeImageTo,fddfb,output_folder,
                           efvr,efhr):


    # Loading configurations
    configParser = configparser.RawConfigParser()   
    configFilePath = r'configuration.txt'
    configParser.read(configFilePath)


    insert_amd_env_vars =  int(configParser.get('COMMON', 'insert_amd_env_vars'))
    HSA_OVERRIDE_GFX_VERSION =  configParser.get('COMMON', 'HSA_OVERRIDE_GFX_VERSION')
    ROCM_PATH =  configParser.get('COMMON', 'ROCM_PATH')

    if(insert_amd_env_vars != 0):
        os.environ["HSA_OVERRIDE_GFX_VERSION"] = HSA_OVERRIDE_GFX_VERSION
        os.environ["ROCM_PATH"] = ROCM_PATH

    backends = [
        'opencv', 
        'ssd', 
        'dlib', 
        'mtcnn', 
        'retinaface', 
        'mediapipe'
    ]

    vidcap = cv2.VideoCapture(absPathVideo)
    framesNo = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_number = random.randint(1, framesNo - 1)
    frameNo=1
    
    absPathFrame = output_folder + "/" + "frame.png"

    vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_number-1)
    success,image = vidcap.read()
    count = 1
    cv2.imwrite(absPathFrame, image)     # save frame as PNG file

    vidcap.release()


    face_objs = DeepFace.extract_faces(img_path = absPathFrame, 
            target_size = (resizeImageTo, resizeImageTo),
        detector_backend = backends[fddfb]
    )

    if(len(face_objs) == 0):
        os.remove(absPathFrame)
        q.put('Error')
        return
    
    # Gets biggest face in image (because images can have multiple faces)
    facial_area = None
    size_one_facial_area_prev = 0
    for face_obj in face_objs:
        one_facial_area = face_obj['facial_area']
        one_facial_area_h = one_facial_area['h']
        one_facial_area_w = one_facial_area['w']
        size_one_facial_area = one_facial_area_h * one_facial_area_w
        if(size_one_facial_area > size_one_facial_area_prev):
            size_one_facial_area_prev = size_one_facial_area
            facial_area = one_facial_area

    crop_img_start_row = facial_area['y'] - int(efvr * facial_area['h'])
    if(crop_img_start_row < 0):
        crop_img_start_row = 0

    crop_img_start_col = facial_area['x'] - int(efhr * facial_area['w'])
    if(crop_img_start_col < 0):
        crop_img_start_col = 0

    crop_img_end_row = facial_area['y'] + int(efvr * facial_area['h'])+facial_area['h']
    if(crop_img_end_row >= len(image)):
        crop_img_end_row = len(image) - 1
    
    crop_img_end_col = facial_area['x'] + int(efhr * facial_area['w'])+facial_area['w']
    if(crop_img_end_col >= len(image[0])):
        crop_img_end_col = len(image[0]) -1

    crop_img = image[crop_img_start_row : crop_img_end_row, 
                    crop_img_start_col :crop_img_end_col]
    
    crop_img = resizeImage(crop_img,resizeImageTo)
    crop_img = make_square(crop_img,resizeImageTo)

    absPathFace = output_folder + "/" + "face.png"

    
    crop_img.save(absPathFace)

    face_analysis_objs = DeepFace.analyze(img_path = absPathFace, 
          actions = ['age', 'gender', 'race'],enforce_detection = False)
    
    embedding_objs = DeepFace.represent(absPathFace)
    embedding = embedding_objs[0]["embedding"]
    
    if(len(face_analysis_objs) == 1):
        gender = face_analysis_objs[0]['dominant_gender']
        ethnicity = face_analysis_objs[0]['dominant_race']
        age = face_analysis_objs[0]['age']
    
    else:
        gender = 'Error'
        ethnicity = 'Error'
        age = 'Error'

    q.put(gender)
    q.put(ethnicity)
    q.put(age)

    os.remove(absPathFrame)

    #embeddingsPickle = pickle.dumps(embedding)
    #print(embedding)

    with open(output_folder + '/' + 'vgg.pickle', 'wb') as handle:
        pickle.dump(embedding, handle)


