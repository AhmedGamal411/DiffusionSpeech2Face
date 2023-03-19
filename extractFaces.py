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

# TODO Make parrallizm

origin_time = time.time()

configParser = configparser.RawConfigParser()   
configFilePath = r'configuration.txt'
configParser.read(configFilePath)

datasetPathVideo =  configParser.get('COMMON', 'datasetPathVideo')
datasetPathFrames =  configParser.get('extractFaces', 'datasetPathFrames')
datasetPathFaces =  configParser.get('extractFaces', 'datasetPathFaces')
efvr =  float(configParser.get('extractFaces', 'expandFaceVerticalRatio'))
efhr =  float(configParser.get('extractFaces', 'expandFaceHorizontalRatio'))
resizeImageTo =  int(configParser.get('extractFaces', 'resizeImageTo'))
p =  float(configParser.get('extractFaces', 'parallelism'))
fddfb =  int(configParser.get('extractFaces', 'faceDetectionDeepFaceBackend'))
pf =  int(configParser.get('extractFaces', 'parallelismFrames'))
datasetPathDatabase =  configParser.get('COMMON', 'datasetPathDatabase') + '/dataset.db'

con = sl.connect(datasetPathDatabase)

print('about to start')


def resizeImage(im, size=resizeImageTo):
    
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

def make_square(im, min_size=resizeImageTo, fill_color=(0, 0, 0, 0)):
    x, y = im.size
    size = max(min_size, x, y)
    new_im = Image.new('RGBA', (size, size), fill_color)
    new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
    return new_im


def extractFrames(absPathVideo,videoId,frameNo):


  vidcap = cv2.VideoCapture(absPathVideo)
  framesNo = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
  random.seed(frameNo)
  frame_number = random.randint(1, framesNo - 1)


  absPathFrame = absPathVideo.replace(datasetPathVideo,datasetPathFrames)
  absPathFrame = os.path.splitext(absPathFrame)[0]
  absPathFrame = absPathFrame + "_frame_" + str(frameNo) + ".png"
  #print(absPathFrame)
  #Create Directory
  pathlib.Path(os.path.dirname(absPathFrame)).mkdir(parents=True, exist_ok=True) 


  vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_number-1)
  success,image = vidcap.read()
  count = 1
  cv2.imwrite(absPathFrame, image)     # save frame as PNG file


  backends = [
    'opencv', 
    'ssd', 
    'dlib', 
    'mtcnn', 
    'retinaface', 
    'mediapipe'
  ]
  face_objs = DeepFace.extract_faces(img_path = absPathFrame, 
        target_size = (resizeImageTo, resizeImageTo),
       detector_backend = backends[fddfb]
  )

  if(len(face_objs) == 0):
     os.remove(absPathFrame)
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


  # TODO if more than one face, get biggest face DONE
  # TODO if no face DONE
  # TODO multiprocessing on single video and on multiple videos ignore
  # TODO AGE, ETHNITHTy & GENDER DONE
  # TODO INSERT AGE,ETHNITHTY AND GENDER INTO DONE
  # TODO multiple frames DONE
  # TODO and multiple faces done
  # TODO DELETE FRAME
  # TODO remove background other file
  # TODO normalize other file\
  # TODO have to take on or more ethnicity
  # TODO Encode ethnicity and gender in system
  


  # Gets a crop of the image that is a little bigger than the image of the face only (hair, chin ... etc)
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
  
  crop_img = resizeImage(crop_img)
  crop_img = make_square(crop_img)


  #face_img = (face_obj[0]['face'])
  
  #new_img=cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)
  #plt.imshow(new_img)
  #plt.show()
  #cv2.imshow('image',new_img)
  #new_img = Image.SAVE

  absPathFace = absPathVideo.replace(datasetPathVideo,datasetPathFaces)
  absPathFace = os.path.splitext(absPathFace)[0]
  absPathFace = absPathFace + "_face_" + str(frameNo) + ".png"
  #print(absPathFace)
  #Create Directory
  pathlib.Path(os.path.dirname(absPathFace)).mkdir(parents=True, exist_ok=True) 


  crop_img.save(absPathFace)

  cur = con.cursor()
  if(frameNo == 1):
    face_analysis_objs = DeepFace.analyze(img_path = absPathFace, 
          actions = ['age', 'gender', 'race'],enforce_detection = False)
    
    if(len(face_analysis_objs) == 1):
        gender = face_analysis_objs[0]['dominant_gender']
        ethnicity = face_analysis_objs[0]['dominant_race']
        age = face_analysis_objs[0]['age']

        sql = ''' UPDATE VIDEO SET AGE = ? , GENDER = ? , ETHNICITY = ? WHERE ID = ?'''
  
        with con:
          
          data = [age,gender,ethnicity,videoId]
          cur.execute(sql, data)
          con.commit()


    

    



  sql = ''' INSERT INTO FACE (FACE_PATH, VIDEO_ID) VALUES(?,?)'''
  
  with con:
    data = [absPathFace,videoId]
    cur.execute(sql, data)
    con.commit()
  
  sql = ''' UPDATE VIDEO SET FACES_PRE = 1 WHERE ID = ?'''
  
  with con:
    data = [videoId]
    cur.execute(sql, data)
    con.commit()
    cur.close()

  os.remove(absPathFrame)



#TODO document conda install -c michael_wild opencv-contrib



#for root, dirs, files in os.walk(datasetPathVideo):
#    for file in files:
#        extractFrames(os.path.join(root, file))

with con:
    index = 0
    cont = True
        
    while(cont):
        data = con.execute("SELECT * FROM VIDEO WHERE FACES_PRE = 0 ORDER BY ID LIMIT "+str(p)+" OFFSET " + str(index))
        cont = False
        for row in data:
          cont=True
          for frameNo in range(1,pf + 1):
            try:
              extractFrames(row[1],row[0],frameNo)
            except:
               pass

        index = index + p

with con:
    data = con.execute("SELECT COUNT(*) FROM FACE")
    for row in data:
        print(str(row) + " FACE FILES INSERTED") 

with con:
    data = con.execute("SELECT * FROM FACE")
    for row in data:
        print(row) 

with con:
    data = con.execute("SELECT * FROM VIDEO")
    for row in data:
        print(row) 

print('----------------------------------------------------------------      FINISHED      -----------------------------------------')



time_interval = time.time() - origin_time
print("took " +  str(time_interval))
