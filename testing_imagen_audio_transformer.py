
import pickle

import os
import configparser
import torch
import numpy as np
from transformers import AutoProcessor, ASTModel
from scipy.io import wavfile
import torch
import pickle


def extract_audio_transformer(q,path_var_len_audio,output_folder):

    processor = AutoProcessor.from_pretrained("MIT/ast-finetuned-audioset-16-16-0.442")
    model = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-16-16-0.442")

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

    samplerate, data = wavfile.read(path_var_len_audio)
    data = data.astype(np.float32)

    inputs = processor(data, sampling_rate=samplerate, return_tensors="pt")

    with torch.no_grad():

        outputs = model(**inputs)

    last_hidden_states = outputs.last_hidden_state


    emb = last_hidden_states.cpu().detach().numpy()

    embeddingsPickle = pickle.dumps(emb)
    


    with open(output_folder + '/' + 'audio_features.pickle', 'wb') as handle:
        pickle.dump(embeddingsPickle, handle, protocol=pickle.HIGHEST_PROTOCOL)
