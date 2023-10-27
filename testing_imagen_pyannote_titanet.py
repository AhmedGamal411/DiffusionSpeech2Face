
import pickle
import openl3
import torch
import torchaudio

import soundfile as sf
from pyannote.audio import Model
from pyannote.audio import Inference
import nemo.collections.asr as nemo_asr
import numpy as np

import os
import configparser
from pyannote.audio import Model
from pyannote.audio import Inference


    
def extract_pyannote_titanet_embeddings_per_row(row,
                                                inference0,inference1,inference2,inference3,
                                                inference4,inference5,speaker_model,
                                                speaker_model2):

    path_var_len_audio = row['path_var_len_audio']

    emb0 = inference0(path_var_len_audio).data
    emb1 = inference1(path_var_len_audio).data
    emb2 = inference2(path_var_len_audio).data
    emb3 = inference3(path_var_len_audio).data
    emb4 = inference4(path_var_len_audio).data
    emb5 = inference5(path_var_len_audio).data

    emb = np.vstack((emb0,emb1))
    emb = np.vstack((emb,emb2))
    emb = np.vstack((emb,emb3))
    emb = np.vstack((emb,emb4))
    emb = np.vstack((emb,emb5))
    emb = emb / 400.0

    embTitaNet = speaker_model.get_embedding(path_var_len_audio)
    c = embTitaNet.detach().cpu().numpy()
    c = c.squeeze()
    c = np.pad(c, (160), 'constant', constant_values=(0))
    c = c * 10.0
    emb = np.vstack((emb,c))


    embSpeakerNet = speaker_model2.get_embedding(path_var_len_audio)
    c = embSpeakerNet.detach().cpu().numpy()
    c = c.squeeze()
    c = np.pad(c, (128), 'constant', constant_values=(0))
    c = c * 1.0
    emb = np.vstack((emb,c))



    embeddingsPickle2 = pickle.dumps(emb)

    return embeddingsPickle2




    

def extract_pyannote_titanet_embeddings(dataGotten,output_folder):

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

    use_auth_token =  configParser.get('extractPyannoteTitaNet', 'use_auth_token')

    model = None
    inference0 = None
    inference1 = None
    inference2 = None
    inference3 = None
    inference4 = None
    inference5 = None
    speaker_model = None



    model = Model.from_pretrained("pyannote/embedding", 
                                use_auth_token=use_auth_token)
    

    inference0 = Inference(model, window="sliding",
                        duration=0.75, step=0.25 ,device=torch.device(0))
    inference1 = Inference(model, window="sliding",
                        duration=1.5, step=0.5 ,device=torch.device(0))
    inference2 = Inference(model, window="sliding",
                        duration=3, step=1 ,device=torch.device(0))
    inference3 = Inference(model, window="sliding",
                        duration=6, step=2 ,device=torch.device(0))
    inference4 = Inference(model, window="sliding",
                        duration=12, step=4 ,device=torch.device(0))
    inference5 = Inference(model, window="sliding",
                        duration=24, step=8 ,device=torch.device(0))

    
    speaker_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained("nvidia/speakerverification_en_titanet_large")
    speaker_model2 = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(model_name="speakerverification_speakernet")

    dataGotten['AUDIO_EMB'] = dataGotten.apply(extract_pyannote_titanet_embeddings_per_row,
                                               args=(inference0,inference1,inference2,inference3,
                                                inference4,inference5,speaker_model,
                                                speaker_model2),axis=1)

    with open(output_folder + '/' + 'dataGotten3.pickle', 'wb') as handle:
        pickle.dump(dataGotten, handle)