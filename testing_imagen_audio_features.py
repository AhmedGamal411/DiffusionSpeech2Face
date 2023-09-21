
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


def extract_audio_features(q,path_var_len_audio,output_folder):


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


    [Fs, x] = audioBasicIO.read_audio_file(path_var_len_audio)
    F, f_names = ShortTermFeatures.feature_extraction(x, Fs, int(0.5*Fs), int(0.3*Fs))


    x3, sr3 = librosa.load(path_var_len_audio)
    lff = librosa.feature.melspectrogram(y=x3, sr=sr3,n_fft=512,hop_length=int(0.3*Fs))


    brick = np.zeros((128-68,79))
    cc = np.vstack((brick,F))
    emb = np.hstack((cc,lff))

    emb= np.transpose(emb)
    # 190 * 128 (note that this is sequenctial 190 timesframes, 128 features)

    embeddingsPickle = pickle.dumps(emb)
    


    with open(output_folder + '/' + 'audio_features.pickle', 'wb') as handle:
        pickle.dump(embeddingsPickle, handle)
