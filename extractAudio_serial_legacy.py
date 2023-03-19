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
start_time = time.time()

configParser = configparser.RawConfigParser()   
configFilePath = r'configuration.txt'
configParser.read(configFilePath)

datasetPathVideo =  configParser.get('COMMON', 'datasetPathVideo')
datasetPathAudio =  configParser.get('extractAudio', 'datasetPathAudio')
p =  configParser.get('extractAudio', 'dbChunk')
ttwbdf =  int(configParser.get('extractAudio', 'time_to_wait_before_deleting_files'))
cuda =  int(configParser.get('COMMON', 'cuda'))
print(datasetPathVideo)

con = sl.connect('dataset.db',timeout=10)

print('about to start')

if(cuda == 0):
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
    classifierLang = EncoderClassifier.from_hparams(source="speechbrain/lang-id-commonlanguage_ecapa", savedir="pretrained_models/lang-id-commonlanguage_ecapa")
else:
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",run_opts={"device":"cuda"})
    classifierLang = EncoderClassifier.from_hparams(source="speechbrain/lang-id-commonlanguage_ecapa", savedir="pretrained_models/lang-id-commonlanguage_ecapa", run_opts={"device":"cuda"})
    
filesToDelete = []

def extractAudio(absPathVideo,rowId):
    absPathAudio = y = absPathVideo.replace(datasetPathVideo,datasetPathAudio)
    absPathAudio = os.path.splitext(absPathAudio)[0]
    absPathAudio_w = absPathAudio
    absPathAudio = absPathAudio + "_audio.wav"

    #Create Directory
    pathlib.Path(os.path.dirname(absPathAudio)).mkdir(parents=True, exist_ok=True) 


    command = "ffmpeg -nostats -loglevel 0 -y -i '" + absPathVideo + "' -acodec pcm_s16le -ab 160k -ac 1 -ar 16000 -vn '" + absPathAudio + "'"
    subprocess.call(command, shell=True)

    audio = AudioSegment.from_file(absPathVideo)
    audio_length_og = math.floor(audio.duration_seconds)
    #print(audio_length_og)
    

   

    audio_length_list = [3,6,12,24]
    for audio_length in audio_length_list:
        path_var_len_audio =  absPathAudio_w + "audio" + str(audio_length) + "s.wav"
        path_var_len_audio_temp =  absPathAudio_w + "audio_temp" + str(audio_length) + "s.wav"

        if(audio_length_og > audio_length):
            # Truncate    

             

            command = "ffmpeg -nostats -loglevel 0 -y -ss 0 -t "+str(audio_length)+" -i \"" + absPathAudio + "\" \"" + path_var_len_audio + "\""
            subprocess.call(command, shell=True)

            # TODO 16khz done
            # TODO speechbrain language identification done 
            # TODO Delete audios done
            # TODO DOCUMENT INSTALL sudo apt install ubuntu-restricted-extras , SPEECHBRAIN, FFMPEG
            # TODO document will create files but delete them at the end
            # TODO CITATION speechbrain AND ECAPA
            # TODO DOCUMENT CODE
            # TODO PARALLELIZE Done
            # TODO TIME TO WAIT AFTER BATCH BEFORE DELETING FILES

        else:
            #print("lesa")
            twoDigitLenStr = f"{audio_length:02}"
            #print(twoDigitLenStr)
            command = "ffmpeg -nostats -loglevel 0 -y -stream_loop -1 -i '" + absPathAudio + "' -t \"00:00:"+twoDigitLenStr+".000\" -codec:a \"aac\" -f \"wav\" -c copy '"+ path_var_len_audio_temp + "'"
            subprocess.call(command, shell=True)
            command = "ffmpeg -nostats -loglevel 0 -y -ss 0 -t "+str(audio_length)+" -i \"" + path_var_len_audio_temp + "\" \"" + path_var_len_audio + "\""
            subprocess.call(command, shell=True)

        
        
        signal, fs = torchaudio.load(path_var_len_audio)
        embeddings = classifier.encode_batch(signal)
        embeddingsPickle = pickle.dumps(embeddings)

        out_prob, score, index, text_lab = classifierLang.classify_file(path_var_len_audio)
        lang = text_lab[0]

        

        sql = ''' UPDATE VIDEO SET AUDIO_PATH = ? WHERE ID = ?'''

        with con:
            cur = con.cursor()
            data = [absPathAudio,rowId]
            cur.execute(sql, data)
            con.commit()



        sql = ''' INSERT INTO AUDIO (VIDEO_ID,AUDIO_LENGTH,SPEAKER_EMB,LANG) VALUES(?,?,?,?)'''

        with con:
            data = [rowId,audio_length,embeddingsPickle,lang]
            cur.execute(sql, data)
            con.commit()
            cur.close()


        filesToDelete.append(absPathAudio)
        filesToDelete.append(path_var_len_audio)
        filesToDelete.append(os.path.basename(path_var_len_audio))
        filesToDelete.append(path_var_len_audio_temp)

        


        
def delFiles():
    for file in filesToDelete:      
        try:
            os.remove(file)
        except OSError:
            pass

    


with con:
    contLoop = True
    offset = 0
    while(contLoop):
        data = con.execute("SELECT * FROM VIDEO LIMIT " + p + " OFFSET " + str(offset))
        contLoop = False
        offset = offset + int(p)
        print("Got chunk of videos from database, preparing to extract audio and extract features")
        for row in data:
            contLoop = True
            extractAudio(row[1],row[0])
        time.sleep(ttwbdf)
        delFiles()
        
print('----------------------------------------------------------------      FINISHED      -----------------------------------------')

with con:
    data = con.execute("SELECT count(*) FROM VIDEO")
    for row in data:
        print("THERE WERE " + str(row) + " VIDEO FILES")   

with con:
    data = con.execute("SELECT count(*) FROM AUDIO")
    for row in data:
        print(str(row) + " AUDIO FILES INSERTED")


print("--- %s seconds ---" % (time.time() - start_time))




