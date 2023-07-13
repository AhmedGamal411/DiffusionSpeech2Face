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

start_time = time.time()    # To measure execution time in seconds

print("PLEASE EDIT configuration.txt BEFORE EXECUTION")
print(".wav files might be generated in path. The program will automatically delete them. If execuetion stops unexpectedly, please delete them yourself")

# Loading configurations
configParser = configparser.RawConfigParser()   
configFilePath = r'configuration.txt'
configParser.read(configFilePath)

datasetPathVideo =  configParser.get('COMMON', 'datasetPathVideo')
datasetPathDatabase =  configParser.get('COMMON', 'datasetPathDatabase') + '/dataset.db'
datasetPathDatabaseAdditional =  configParser.get('COMMON', 'datasetPathDatabase') + '/datasetAdditional.db'
cpus =  int(configParser.get('COMMON', 'cpus'))

datasetPathAudio =  configParser.get('extractAudioFeatures', 'datasetPathAudio')

p =  configParser.get('extractAudioFeatures', 'dbChunk')
ttwbdf =  int(configParser.get('extractAudioFeatures', 'time_to_wait_before_deleting_files'))

print("Video dataset at " + datasetPathVideo )

import os
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
os.environ["ROCM_PATH"] = "/opt/rocm"

con = sl.connect(datasetPathDatabase)  # Connection to databases

print('------------------- ABOUT TO START --------------------')
#TODO what if two files have the same name in the same batch


from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import ShortTermFeatures
import librosa
 
def extractAudio(rows):
    con2 = sl.connect(datasetPathDatabase)
    conAdditional = sl.connect(datasetPathDatabaseAdditional)  # Connection to databases
    #print(rows)

    for row in rows:
        #print(row)
        absPathVideo = row[1]   # for this one video
        rowId = row[0]          # id in database

        absPathAudio = y = absPathVideo.replace(datasetPathVideo,datasetPathAudio)  # for this one audio
        absPathAudio = os.path.splitext(absPathAudio)[0]
        absPathAudio_w = absPathAudio   # without the end
        absPathAudio = absPathAudio + "_audio.wav"  # full path to extracted audio from the video

        #Create Directory
        pathlib.Path(os.path.dirname(absPathAudio)).mkdir(parents=True, exist_ok=True) 

        # Extract audio monochannel and with 16khz and put it in absPathAudio
        command = "ffmpeg -nostats -loglevel 0 -y -i '" + absPathVideo + "' -acodec pcm_s16le -ab 160k -ac 1 -ar 16000 -vn '" + absPathAudio + "'"
        subprocess.call(command, shell=True)


        # Get original duration of video
        audio = AudioSegment.from_file(absPathVideo)
        audio_length_og = math.floor(audio.duration_seconds)
        #print(audio_length_og)
        

    
        # Will either truncate or loop the original video to reach audio_length (3,6,12 or 24)
        audio_length_list = [24]
        for audio_length in audio_length_list:
            path_var_len_audio =  absPathAudio_w + "audio" + str(audio_length) + "s.wav"    # path to the variable length audio
            path_var_len_audio_temp =  absPathAudio_w + "audio_temp" + str(audio_length) + "s.wav"  # path to a temp version of the variable length audio

            if(audio_length_og > audio_length):
                # Truncate    

                command = "ffmpeg -nostats -loglevel 0 -y -ss 0 -t "+str(audio_length)+" -i \"" + absPathAudio + "\" \"" + path_var_len_audio + "\""
                subprocess.call(command, shell=True)


            else:
                # Loop then truncaate
                #print("lesa")
                twoDigitLenStr = f"{audio_length:02}"
                #print(twoDigitLenStr)
                command = "ffmpeg -nostats -loglevel 0 -y -stream_loop -1 -i '" + absPathAudio + "' -t \"00:00:"+twoDigitLenStr+".000\" -codec:a \"aac\" -f \"wav\" -c copy '"+ path_var_len_audio_temp + "'"
                subprocess.call(command, shell=True)
                command = "ffmpeg -nostats -loglevel 0 -y -ss 0 -t "+str(audio_length)+" -i \"" + path_var_len_audio_temp + "\" \"" + path_var_len_audio + "\""
                subprocess.call(command, shell=True)

            # Extract audio embeddings
            #emb = []
            #queue = Queue()
            #proc = Process(target=openSb.extractOpenL3Subprocess, args=(audio,sr,queue,))   # spawn a process
            #proc.start()
            #proc.join()
            #emb = queue.get()

            [Fs, x] = audioBasicIO.read_audio_file(path_var_len_audio)
            F, f_names = ShortTermFeatures.feature_extraction(x, Fs, int(0.5*Fs), int(0.3*Fs))
            
            x3, sr3 = librosa.load(path_var_len_audio)
            lff = librosa.feature.melspectrogram(y=x3, sr=sr3,n_fft=512,hop_length=int(0.3*Fs))

            brick = np.zeros((128-68,79))
            cc = np.vstack((brick,F))
            emb = np.hstack((cc,lff))

            emb= np.transpose(emb)
            # 190 * 128 (note that this is sequenctial 190 timesframes, 128 features)

            #print(emb.shape)

            embeddingsPickle = pickle.dumps(emb)
            #update audio embeddings into database
            sql = ''' INSERT INTO AUDIO (VIDEO_ID,AUDIO_LENGTH,AUDIO_FEATURES) VALUES(?,?,?)'''
            #print(emb.shape)
            
            cur = conAdditional.cursor()
            data = [rowId,audio_length,embeddingsPickle]
            cur.execute(sql, data)
            conAdditional.commit()
            cur.close()
            #print(emb.shape)

            # Will delete those files after a little bit
            ftd = [absPathAudio,path_var_len_audio,os.path.basename(path_var_len_audio),path_var_len_audio_temp]
            tDelete = Thread(target=delFiles, args=(ftd,))   # spawn a process
            tDelete.start()
            
        sql = '''UPDATE VIDEO SET AUDIO_PRE = 3 WHERE ID = ?'''
        cur2 = con2.cursor()
        data = [rowId]
        cur2.execute(sql, data)
        con2.commit()
        cur2.close()
        #print(emb.shape)


           
            
# Function to delete audio temp files
def delFiles(filesToDelete):
    time.sleep(ttwbdf)  # wait a bit
    for file in filesToDelete:  
        try:
            os.remove(file)
        except OSError:
            pass
        



# TODO Better display of progress and handling of exceptions
contLoop = True # Flag to continue to get chunks of videos from database
offset = 0
while(contLoop):
    data = con.execute("SELECT * FROM VIDEO WHERE AUDIO_PRE = 2 AND FACES_PRE = 1 ORDER BY ID ASC LIMIT " + p + " OFFSET " + str(offset))
    contLoop = False
    offset = offset + int(p)
    print("Got chunk of videos from database. Extracting audio and audio features...")
    # TODO write time
    
    #print(data.fetchall())
    dataGotten = data.fetchall()
    rowsPerProcess = math.ceil(len(dataGotten) / cpus)  # Will spawn no. of processes = cpus, each will get rows = rowsperprocess
    procs = []
    while(len(dataGotten) > 0):
        rows=dataGotten[:rowsPerProcess]    # rows to be sent to a process
        dataGotten = dataGotten[rowsPerProcess:]    # Deletes the rows that are going to be sent from dataGotten
        #print(rows)
        contLoop = True # Continue to get data from database since data length is not 0
        #extractAudio(rows)
        proc = Process(target=extractAudio, args=(rows,))   # spawn a process
        procs.append(proc)
        proc.start()
    for proc in procs:  # wait for all processes to finish
        proc.join()
    
print('----------------------------------------------------------------      FINISHED      -----------------------------------------')

print("--- %s seconds ---" % (time.time() - start_time))