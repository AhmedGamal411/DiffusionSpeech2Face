
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


def extract_pyannote_titanet_embeddings(q,path_var_len_audio,audio_embs,audio_length,openl3_mode,output_folder):

    print('a')
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


    print('b')
    model = None
    inference0 = None
    inference1 = None
    inference2 = None
    inference3 = None
    inference4 = None
    inference5 = None
    speaker_model = None

    if(audio_embs == 'openl3'):
        model = openl3.models.load_audio_embedding_model(input_repr="mel128", content_type="env",
                                                    embedding_size=512)
    elif(audio_embs == 'wav2vec'):
        torch.random.manual_seed(0)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        bundle = torchaudio.pipelines.WAV2VEC2_XLSR_300M
        model = bundle.get_model().to(device)

    elif(audio_embs == 'pyannoteTitaNet'):
        print('c')
        from pyannote.audio import Model
        model = Model.from_pretrained("pyannote/embedding", 
                                    use_auth_token=use_auth_token)
        from pyannote.audio import Inference

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
        print('d')
        import nemo.collections.asr as nemo_asr
        speaker_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained("nvidia/speakerverification_en_titanet_large")
        speaker_model2 = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(model_name="speakerverification_speakernet")
        print('e')

    else:
        raise ValueError('''Specify an audio embedding scheme from the available options in the 
                        configuration file''')


    if(audio_embs == 'openl3'):
        audio, sr = sf.read(path_var_len_audio)

        hop_size = -1
        if(openl3_mode == 'imagen'):
            hop_size = audio_length/250
        elif(openl3_mode == 'stable'):
            hop_size = 24/50
        else:
            raise ValueError('openl3_mode in configuration must either be stable or imagen') 
        
        emb, ts = openl3.get_audio_embedding(audio, sr,hop_size=hop_size,verbose=0,model=model)

        embeddingsPickle2 = pickle.dumps(emb)
    elif(audio_embs == 'wav2vec'):
        waveform, sample_rate = torchaudio.load(path_var_len_audio)
        waveform = waveform.to(device)

        if sample_rate != bundle.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)
        with torch.inference_mode():
            features, _ = model.extract_features(waveform)


        tensor =  torch.empty(( 0,features[0].shape[1],768), dtype=torch.float32).cuda()
        for x in features:
            tensor = torch.vstack((tensor,x))

        array = tensor.detach().cpu().numpy()
        array = (array/20)

        emb = np.mean(array, axis=0)


        embeddingsPickle2 = pickle.dumps(emb)

    elif(audio_embs == 'pyannoteTitaNet'):
        #path_var_len_audio = "v.mp3"
        print('f')
        emb0 = inference0(path_var_len_audio).data
        emb1 = inference1(path_var_len_audio).data
        emb2 = inference2(path_var_len_audio).data
        emb3 = inference3(path_var_len_audio).data
        emb4 = inference4(path_var_len_audio).data
        emb5 = inference5(path_var_len_audio).data
        print('g')

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
        print('h')
    

        embSpeakerNet = speaker_model2.get_embedding(path_var_len_audio)
        c = embSpeakerNet.detach().cpu().numpy()
        c = c.squeeze()
        c = np.pad(c, (128), 'constant', constant_values=(0))
        c = c * 1.0
        emb = np.vstack((emb,c))

        print('i')

        embeddingsPickle2 = pickle.dumps(emb)
        #q.put(embeddingsPickle2)

        with open(output_folder + '/' + 'audio_emb_pyannote_titanet.pickle', 'wb') as handle:
            pickle.dump(embeddingsPickle2, handle)

        print('j')