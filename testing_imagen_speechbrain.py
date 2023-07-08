import speechbrain as sb
from speechbrain.pretrained import EncoderClassifier
import torchaudio
import pickle

import os
import configparser


def extract_speechbrain_embeddings(q,path_var_len_audio):

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


    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
    # Extract speaker embeddings
    signal, fs = torchaudio.load(path_var_len_audio)
    embeddings = classifier.encode_batch(signal)
    embeddingsPickle = pickle.dumps(embeddings.cpu().detach().numpy())
    q.put(embeddingsPickle)