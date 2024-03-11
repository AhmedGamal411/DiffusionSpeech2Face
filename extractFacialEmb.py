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


import os
import pathlib
import configparser
import sqlite3 as sl
import math
import pickle
import shutil
import time
import itertools
import torch
import numpy as np
import random
from deepface import DeepFace


import os

start_time = time.time()    # To measure execution time in seconds

print("PLEASE EDIT configuration.txt BEFORE EXECUTION")

# Loading configurations
configParser = configparser.RawConfigParser()   
configFilePath = r'configuration.txt'
configParser.read(configFilePath)

datasetPathDatabase =  configParser.get('COMMON', 'datasetPathDatabase') + '/dataset.db'
datasetPathDatabaseAdditionalFE = configParser.get('COMMON', 'datasetPathDatabase') + '/datasetAdditionalFE.db'
cpus =  int(configParser.get('COMMON', 'cpus'))


p =  configParser.get('extractFacialEmb', 'dbChunk')


import os
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
os.environ["ROCM_PATH"] = "/opt/rocm"

con = sl.connect(datasetPathDatabase)  # Connection to databases
conAdditionalFE = sl.connect(datasetPathDatabaseAdditionalFE)  # Connection to databases

print('------------------- ABOUT TO START --------------------')

 
def extractFacialEmbeddings(rows):
    for row in rows:
        #print(row)
        absPathFace = row[2]   # for this one video
        videoId = row[0]          # id in database
        faceId = row[1]

        embedding_objs = DeepFace.represent(absPathFace,enforce_detection=False)
        embedding = embedding_objs[0]["embedding"]
        embeddingsPickle = pickle.dumps(embedding)

        #update audio embeddings into database
        sql = ''' INSERT INTO FACE_EMB (VIDEO_ID,FACE_ID,FACE_EMB) VALUES(?,?,?)'''
        
        curFE = conAdditionalFE.cursor()
        data = [videoId,faceId,embeddingsPickle]
        curFE.execute(sql, data)
        conAdditionalFE.commit()
        curFE.close()

        sql = '''UPDATE VIDEO SET FACES_PRE = 3 WHERE ID = ?'''
        cur = con.cursor()
        data = [videoId]
        cur.execute(sql, data)
        con.commit()
        cur.close()            

contLoop = True # Flag to continue to get chunks of videos from database
while(contLoop):
    data = con.execute("""SELECT V.ID,F.ID,F.FACE_PATH FROM VIDEO V 
      INNER JOIN FACE F ON F.ID = (select ID from FACE f2 where f2.video_id = v.ID ORDER By ID limit 1)
      WHERE V.FACES_PRE IN (1,2) AND AUDIO_PRE IN (3,4) LIMIT """ + p )
    contLoop = False
    #print("Got chunk of FACES from database. Extracting VGG Face features...")
    dataGotten = data.fetchall()
    if(len(dataGotten) > 0):
        contLoop = True # Continue to get data from database since data length is not 0
        extractFacialEmbeddings(dataGotten)

    
print('----------------------------------------------------------------      FINISHED      -----------------------------------------')

print("--- %s seconds ---" % (time.time() - start_time))