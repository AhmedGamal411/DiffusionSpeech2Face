import subprocess
import os
import pathlib
import configparser
import sqlite3 as sl
import speechbrain as sb
import torchaudio
from speechbrain.pretrained import EncoderClassifier
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


start_time = time.time()    # To measure execution time in seconds

print("PLEASE EDIT configuration.txt BEFORE EXECUTION")
print(".wav files might be generated in path. The program will automatically delete them. If execuetion stops unexpectedly, please delete them yourself")

# Loading configurations
configParser = configparser.RawConfigParser()   
configFilePath = r'configuration.txt'
configParser.read(configFilePath)

datasetPathVideo =  configParser.get('COMMON', 'datasetPathVideo')
datasetPathAudio =  configParser.get('extractAudio', 'datasetPathAudio')
p =  configParser.get('extractAudio', 'dbChunk')
ttwbdf =  int(configParser.get('extractAudio', 'time_to_wait_before_deleting_files'))
cuda =  int(configParser.get('COMMON', 'cuda'))
cpus =  int(configParser.get('COMMON', 'cpus'))
datasetPathDatabase =  configParser.get('COMMON', 'datasetPathDatabase') + '/dataset.db'
# TODO dataset.db in configuration too

print("Video dataset at " + datasetPathVideo )
print("Number of cpus to use for multiprocessing : ", cpus)

if(cuda == 0):
    print("Not using cuda")
else:
    print("Will try to use cuda, if no cuda is present please set cuda = 0 in configuration.txt")

con = sl.connect(datasetPathDatabase)  # Connection to databases

print('------------------- ABOUT TO START --------------------')
#TODO what if two files have the same name in the same batch

 
def extractAudio(rows):
    #print(rows)

    con2 = sl.connect(datasetPathDatabase)

    # Embedding extractor and language classifier
    if(cuda == 0):
        classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
        classifierLang = EncoderClassifier.from_hparams(source="speechbrain/lang-id-commonlanguage_ecapa", savedir="pretrained_models/lang-id-commonlanguage_ecapa")
    else:
        classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",run_opts={"device":"cuda"})
        classifierLang = EncoderClassifier.from_hparams(source="speechbrain/lang-id-commonlanguage_ecapa", savedir="pretrained_models/lang-id-commonlanguage_ecapa", run_opts={"device":"cuda"})

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
        audio_length_list = [6,12,24]
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

            # Extract speaker embeddings
            signal, fs = torchaudio.load(path_var_len_audio)
            embeddings = classifier.encode_batch(signal)
            embeddingsPickle = pickle.dumps(embeddings.cpu().detach().numpy()) # pickle embeddings to put in database

            # Language classification
            out_prob, score, index, text_lab = classifierLang.classify_file(path_var_len_audio)
            lang = text_lab[0]

            # Insert speaker embeddings and language into database
            sql = ''' INSERT INTO AUDIO (VIDEO_ID,AUDIO_LENGTH,SPEAKER_EMB,LANG) VALUES(?,?,?,?)'''

            
            cur = con2.cursor()
            data = [rowId,audio_length,embeddingsPickle,lang]
            cur.execute(sql, data)
            con2.commit()

             # Will delete those files after a little bit
            ftd = [absPathAudio,path_var_len_audio,os.path.basename(path_var_len_audio),path_var_len_audio_temp]
            tDelete = Thread(target=delFiles, args=(ftd,))   # spawn a process
            tDelete.start()
           

        sql = '''UPDATE VIDEO SET AUDIO_PRE = 1 WHERE ID = ?'''

        data = [rowId]
        cur.execute(sql, data)
        con2.commit()
        cur.close()


            
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
    data = con.execute("SELECT * FROM VIDEO WHERE AUDIO_PRE = 0 ORDER BY ID ASC LIMIT " + p + " OFFSET " + str(offset))
    contLoop = False
    offset = offset + int(p)
    print("Got chunk of videos from database. Extracting audio and features...")
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
        proc = Process(target=extractAudio, args=(rows,))   # spawn a process
        procs.append(proc)
        proc.start()
    for proc in procs:  # wait for all processes to finish
        proc.join()
    
print('----------------------------------------------------------------      FINISHED      -----------------------------------------')

with con:
    data = con.execute("SELECT count(*) FROM VIDEO")
    for row in data:
        print("THERE WERE " + str(row) + " VIDEO FILES")   

with con:
    data = con.execute("SELECT count(*) FROM AUDIO")
    for row in data:
        print(str(row) + " AUDIO FILES PRESENT IN DATABASE")

print("--- %s seconds ---" % (time.time() - start_time))





