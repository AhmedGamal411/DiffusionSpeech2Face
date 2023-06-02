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
import soundfile as sf
from multiprocessing import Queue
from multiprocessing import Pool
import openl3


start_time = time.time()    # To measure execution time in seconds

print("PLEASE EDIT configuration.txt BEFORE EXECUTION")
print(".wav files might be generated in path. The program will automatically delete them. If execuetion stops unexpectedly, please delete them yourself")

# Loading configurations
configParser = configparser.RawConfigParser()   
configFilePath = r'configuration.txt'
configParser.read(configFilePath)

datasetPathVideo =  configParser.get('COMMON', 'datasetPathVideo')
datasetPathDatabase =  configParser.get('COMMON', 'datasetPathDatabase') + '/dataset.db'
cpus =  int(configParser.get('COMMON', 'cpus'))

datasetPathAudio =  configParser.get('extractOpenL3', 'datasetPathAudio')
openl3_mode =  configParser.get('extractOpenL3', 'openl3_mode')

p =  configParser.get('extractOpenL3', 'dbChunk')
ttwbdf =  int(configParser.get('extractOpenL3', 'time_to_wait_before_deleting_files'))

print("Video dataset at " + datasetPathVideo )
print("Extracting for mode " + openl3_mode)

model = openl3.models.load_audio_embedding_model(input_repr="mel128", content_type="music",
                                                embedding_size=512)


con = sl.connect(datasetPathDatabase)  # Connection to databases
print('------------------- ABOUT TO START --------------------')
#TODO what if two files have the same name in the same batch

 
def extractAudio(rows):
    con2 = sl.connect(datasetPathDatabase)
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

            # Extract audio embeddings
            audio, sr = sf.read(path_var_len_audio)
            #emb = []
            #queue = Queue()
            #proc = Process(target=openSb.extractOpenL3Subprocess, args=(audio,sr,queue,))   # spawn a process
            #proc.start()
            #proc.join()
            #emb = queue.get()
            hop_size = -1
            if(openl3_mode == 'imagen'):
                hop_size = audio_length/250
            elif(openl3_mode == 'stable'):
                hop_size = 24/50
            else:
                raise ValueError('openl3_mode in configuration must either be stable or imagen') 


            #p = Pool(1)
            #emb = p.apply(openSb.extractOpenL3Subprocess, args=(audio,hop_size,sr,))

            emb, ts = openl3.get_audio_embedding(audio, sr,hop_size=hop_size,verbose=0,model=model)

            #emb, ts = openl3.get_audio_embedding(audio, sr,embedding_size=512,hop_size=audio_length/50,verbose=0)
            #signal, fs = torchaudio.load(path_var_len_audio)
            #embeddings = classifier.encode_batch(signal)
            #embeddingsPickle = pickle.dumps(embeddings.cpu().detach().numpy()) # pickle embeddings to put in database
            #print(emb)
            #print(emb.min())
            #print(emb.max())
            #print(audio_length)
            #print(emb.shape)
            #print(emb)
            embeddingsPickle = pickle.dumps(emb)


            # update audio embeddings into database
            openl3_mode_db = ''
            if(openl3_mode == 'imagen'):
                openl3_mode_db = '2'
            sql = ''' UPDATE AUDIO SET ''' + '''AUDIO_EMB'''+openl3_mode_db+''' = ? WHERE VIDEO_ID = ? AND AUDIO_LENGTH = ?'''

            
            cur = con2.cursor()
            data = [embeddingsPickle,rowId,audio_length]
            cur.execute(sql, data)
            con2.commit()

             # Will delete those files after a little bit
            ftd = [absPathAudio,path_var_len_audio,os.path.basename(path_var_len_audio),path_var_len_audio_temp]
            tDelete = Thread(target=delFiles, args=(ftd,))   # spawn a process
            tDelete.start()

        sql = '''UPDATE VIDEO SET AUDIO_PRE = 2 WHERE ID = ?'''

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
    data = con.execute("SELECT * FROM VIDEO WHERE AUDIO_PRE = 1 AND FACES_PRE = 1 ORDER BY ID ASC LIMIT " + p + " OFFSET " + str(offset))
    contLoop = False
    offset = offset + int(p)
    print("Got chunk of videos from database. Extracting audio and audio embeddings...")
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
        extractAudio(rows)
        #proc = Process(target=extractAudio, args=(rows,))   # spawn a process
        #procs.append(proc)
        #proc.start()
    #for proc in procs:  # wait for all processes to finish
        #proc.join()
    
print('----------------------------------------------------------------      FINISHED      -----------------------------------------')

print("--- %s seconds ---" % (time.time() - start_time))