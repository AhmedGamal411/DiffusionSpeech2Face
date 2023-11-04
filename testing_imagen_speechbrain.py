import speechbrain as sb
from speechbrain.pretrained import EncoderClassifier
import torchaudio
import pickle

import os
import configparser
import pandas as pd
import torch

def extract_speechbrain_embeddings(dataGotten,output_folder):
    
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


    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",run_opts={"device":"cuda"})
    classifierLang = EncoderClassifier.from_hparams(source="speechbrain/lang-id-commonlanguage_ecapa", savedir="pretrained_models/lang-id-commonlanguage_ecapa", run_opts={"device":"cuda"})
    

    def torch_audio_load(row):
        print(row['ID'])
        path_var_len_audio = row['path_var_len_audio']
        signal, fs = torchaudio.load(path_var_len_audio)
        embeddings = classifier.encode_batch(signal)
        embeddingsPickle = pickle.dumps(embeddings.cpu().detach().numpy())
        return embeddingsPickle
    
        
    
    dataGotten['SPEAKER_EMB'] = dataGotten.apply(torch_audio_load,args=(),axis=1)

    
    def get_lang(row):
        print(row['ID'])
        out_prob, score, index, text_lab = classifierLang.classify_file(row['path_var_len_audio'])
        lang = text_lab[0]
        return lang
    dataGotten['caption_l'] = dataGotten.apply(get_lang,args=(),axis=1)

    with open(output_folder + '/' + 'df_data2.pickle', 'wb') as handle:
        pickle.dump(dataGotten, handle)