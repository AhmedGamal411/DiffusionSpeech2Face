
import pickle

import os
import configparser
import numpy as np
from scipy.io import wavfile
import pickle
import soundfile as sf
import configparser
import os
import configparser
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor,HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
import tensorflow as tf


# Loading configurations
configParser = configparser.RawConfigParser()   
configFilePath = r'configuration.txt'
configParser.read(configFilePath)

REQUIRED_SAMPLE_RATE = 16000
model_path =  configParser.get('train_s2fe_random_forest', 'model_path')
cpus_reg =  int(configParser.get('train_s2fe_random_forest', 'cpus_reg'))

with open(model_path, 'rb') as inp:  # 'rb' for reading in binary mode
    model = pickle.load(inp)

print('--------------------------- MODEL LOADED -----------------------------')


def read_audio_file(file_path):
    with open(file_path, "rb") as f:
        audio_wave, sample_rate = sf.read(f)
    if sample_rate != REQUIRED_SAMPLE_RATE:
        raise ValueError(
            f"sample rate (={sample_rate}) of your files must be {REQUIRED_SAMPLE_RATE}"
        )
    audio_wave = audio_wave
    return audio_wave

def get_fft(audio_wave):
    audio_wave = tf.constant(audio_wave, dtype=tf.float64)

    #audio = tf.squeeze(audio_wave, axis=-1)
    fft = tf.signal.fft(
        tf.cast(tf.complex(real=audio_wave, imag=tf.zeros_like(audio_wave)), tf.complex64)
    )
    #fft = tf.expand_dims(fft, axis=-1)
    #fft = tf.squeeze(fft, axis=1)
    #print()
    fft = fft[0:(audio_wave.shape[0] // 2)]
    fft = tf.math.abs(fft)
    fft = fft.numpy()
    return fft

def extract_face_emb_from_speech_per_row(row):
    print(row['id'])

    audio_wave = read_audio_file(row['path_var_len_audio'])
    #print('audio_wave')
 
    audio_wave = get_fft(audio_wave)
    audio_wave = np.expand_dims(audio_wave,0)
    #print(audio_wave.shape)
    
    y_pred = model.predict(audio_wave)

    print(y_pred)
    embeddingsPickle = pickle.dumps(y_pred)

    return embeddingsPickle









def extract_face_emb_from_speech(dataGotten,output_folder):

    #print(dataGotten)
    configParser = configparser.RawConfigParser()   
    configFilePath = r'configuration.txt'
    configParser.read(configFilePath)

    insert_amd_env_vars =  int(configParser.get('COMMON', 'insert_amd_env_vars'))
    HSA_OVERRIDE_GFX_VERSION =  configParser.get('COMMON', 'HSA_OVERRIDE_GFX_VERSION')
    ROCM_PATH =  configParser.get('COMMON', 'ROCM_PATH')

    if(insert_amd_env_vars != 0):
        os.environ["HSA_OVERRIDE_GFX_VERSION"] = HSA_OVERRIDE_GFX_VERSION
        os.environ["ROCM_PATH"] = ROCM_PATH

    
    dataGotten['FACE_EMB_FROM_SPEECH'] = dataGotten.apply(extract_face_emb_from_speech_per_row,args=(),axis=1)

    with open(output_folder + '/' + 'df_data_face_emb.pickle', 'wb') as handle:
        pickle.dump(dataGotten, handle)



