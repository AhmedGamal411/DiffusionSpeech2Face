import os
import configparser

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

import subprocess
import os
import pathlib
import configparser
import sqlite3 as sl
import torchaudio
import cv2
from pydub import AudioSegment
import math
import pickle
import shutil
import time
import multiprocessing
from multiprocessing import Process,Queue
import itertools
from threading import Thread
from multiprocessing import Queue
from multiprocessing import Pool
import torch
import numpy as np
from PIL import Image,ImageFilter
import random
from transformers import ViTImageProcessor, ViTModel
#from tensorflow.keras.applications.vgg19 import VGG19
#from tensorflow.keras.preprocessing import image
#from tensorflow.keras.applications.vgg19 import preprocess_input

import os

start_time = time.time()    # To measure execution time in seconds

print("PLEASE EDIT configuration.txt BEFORE EXECUTION")

# Loading configurations
configParser = configparser.RawConfigParser()   
configFilePath = r'configuration.txt'
configParser.read(configFilePath)

datasetPathFaces =  configParser.get('extractFaces', 'datasetPathFaces')
datasetPathDatabase =  configParser.get('COMMON', 'datasetPathDatabase') + '/dataset.db'
datasetPathDatabaseFacesBlurred =  configParser.get('COMMON', 'datasetPathDatabase') + '/datasetFacesBlurred.db'
cpus =  int(configParser.get('COMMON', 'cpus'))


p =  configParser.get('extractVggBlurred', 'dbChunk')

boxBlurMin =  int(configParser.get('extractVggBlurred', 'boxBlurMin'))
boxBlurMax =  int(configParser.get('extractVggBlurred', 'boxBlurMax'))

gaussianBlurMin =  int(configParser.get('extractVggBlurred', 'gaussianBlurMin'))
gaussianBlurMax =  int(configParser.get('extractVggBlurred', 'gaussianBlurMax'))

import os
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
os.environ["ROCM_PATH"] = "/opt/rocm"

con = sl.connect(datasetPathDatabase)  # Connection to databases

print('------------------- ABOUT TO START --------------------')


#model = VGG19(weights='imagenet', include_top=False)
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
 
def extractVggBlurred(rows):
    con2 = sl.connect(datasetPathDatabase)
    conAdditional = sl.connect(datasetPathDatabaseFacesBlurred)  # Connection to databases
    #print(len(rows))

    for row in rows:
        #print(row)
        absPathFace = row[1]   # for this one video
        rowId = row[0]          # id in database
        videoId = row[2]          # id in database

        im = Image.open(absPathFace).convert('RGB')

        blurOption = random.randint(1, 3)

        if(blurOption==1):
            imBlurred = im.filter(ImageFilter.BoxBlur(random.randint(boxBlurMin, boxBlurMax))) # 4 to 14
        elif(blurOption==2):
            imBlurred = im.filter(ImageFilter.GaussianBlur(random.randint(gaussianBlurMin, gaussianBlurMax))) # 2 to 6
        elif(blurOption==3):
            imBlurred = im.filter(ImageFilter.BoxBlur(random.randint(int(boxBlurMin/2),int(boxBlurMax/2)))).filter(
                ImageFilter.GaussianBlur(random.randint(int(gaussianBlurMin/2),int(gaussianBlurMax/2)))) 

        #directory = "extractVggBlurred"
        #picFile = directory + "/pic.png"
        #if not os.path.exists(directory):
        #    os.makedirs(directory)

        #imBlurred.save(picFile)

        #img_path = picFile
        inputs = processor(images=imBlurred, return_tensors="pt")
        outputs = model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        features = last_hidden_states.cpu().detach().numpy()
        #print(last_hidden_states.cpu().detach().numpy().min()) #(1,197,768) max=1.23 min=-1.23
        
        #img = image.load_img(img_path, target_size=(224, 224))
        #x = image.img_to_array(img)
        #x = np.expand_dims(x, axis=0)
        #x = preprocess_input(x)

        #features = model.predict(x) #(1,7,7,512)
        #features = features.reshape(-1, features.shape[-1]) #(49,512)


        im.close()
        imBlurred.close()
        #mg.close()
        #os.remove(picFile)
        #print(features.shape) #(49, 512)
        #print("max:" + str(features.max())) #max is 72
        #print("min:" + str(features.min())) #min is 0
        embeddingsPickle = pickle.dumps(features)

        #update audio embeddings into database
        sql = ''' INSERT INTO FACES_BLURRED (FACE_ID,VIDEO_ID,BLURRED_FACE_EMB) VALUES(?,?,?)'''
        
        cur = conAdditional.cursor()
        data = [rowId,videoId,embeddingsPickle]
        cur.execute(sql, data)
        conAdditional.commit()
        cur.close()
        #print(emb.shape)

        sql = '''UPDATE FACE SET VGG_BLURRED = 1 WHERE ID = ?'''
        cur2 = con2.cursor()
        data = [rowId]
        cur2.execute(sql, data)
        con2.commit()
        cur2.close()
        #print(emb.shape)
            



        data = con.execute("SELECT FACES_PRE FROM VIDEO WHERE ID = " + str(videoId))
        facesPre = data.fetchall()
        facesPre = (facesPre[0][0])
        if(facesPre == 1 ):
            sql = '''UPDATE VIDEO SET FACES_PRE = 2 WHERE ID = ?'''
            cur2 = con2.cursor()
            data = [videoId]
            cur2.execute(sql, data)
            con2.commit()
            cur2.close()



           
            


#import time
# TODO Better display of progress and handling of exceptions
contLoop = True # Flag to continue to get chunks of videos from database
while(contLoop):
    #time.sleep(10)
    data = con.execute("SELECT * FROM FACE WHERE VGG_BLURRED IS NULL OR VGG_BLURRED=0 ORDER BY ID ASC LIMIT " + p )
    contLoop = False
    print("Got chunk of FACES from database. Extracting VGG19 features...")
    # TODO write time
    #print(data.fetchall())
    dataGotten = data.fetchall()
    #rowsPerProcess = math.ceil(len(dataGotten) / cpus)  # Will spawn no. of processes = cpus, each will get rows = rowsperprocess
    #procs = []
    if(len(dataGotten) > 0):
        #rows=dataGotten[:rowsPerProcess]    # rows to be sent to a process
        #dataGotten = dataGotten[rowsPerProcess:]    # Deletes the rows that are going to be sent from dataGotten
        #print(rows)
        contLoop = True # Continue to get data from database since data length is not 0
        extractVggBlurred(dataGotten)
        #proc = Process(target=extractVggBlurred, args=(rows,))   # spawn a process
        #procs.append(proc)
        #proc.start()
    #for proc in procs:  # wait for all processes to finish
        #proc.join()
    
print('----------------------------------------------------------------      FINISHED      -----------------------------------------')

print("--- %s seconds ---" % (time.time() - start_time))