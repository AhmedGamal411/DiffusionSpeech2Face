
import pickle

import os
import configparser
import numpy as np
from scipy.io import wavfile
import pickle
import soundfile as sf

import tensorflow as tf
import tensorflow_hub as hub
from wav2vec2 import Wav2Vec2Config


REQUIRED_SAMPLE_RATE = 16000
FACE_EMBEDDING_SIZE = 2622
AUDIO_MAX_LEN = 246000


configParser = configparser.RawConfigParser()   
configFilePath = r'configuration.txt'
configParser.read(configFilePath)
model_weights_path =  configParser.get('finetune_wav2vec2', 'model_weights_path')
BATCH_SIZE = int(configParser.get('finetune_wav2vec2', 'batch_size'))

print('--------------------------- LOADING MODEL -----------------------------')

config = Wav2Vec2Config()
pretrained_layer = hub.KerasLayer("https://tfhub.dev/vasudevgupta7/wav2vec2/1", trainable=True)
inputs = tf.keras.Input(shape=(AUDIO_MAX_LEN,))
hidden_states = pretrained_layer(inputs)
pooled_output = tf.keras.layers.AveragePooling1D(pool_size=50)(hidden_states)
flatten_output = tf.keras.layers.Flatten()(pooled_output)
outputs = tf.keras.layers.Dense(FACE_EMBEDDING_SIZE,activation='linear')(flatten_output)
model = tf.keras.Model(inputs=inputs, outputs=outputs)
model(tf.random.uniform(shape=(BATCH_SIZE, AUDIO_MAX_LEN)))
model.summary()
LEARNING_RATE = 5e-5


#loss_fn = CTCLoss(config, (BATCH_SIZE, AUDIO_MAX_LEN), division_factor=BATCH_SIZE)
loss_fn = tf.keras.losses.MeanAbsoluteError()
optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)
model.load_weights(model_weights_path)
model.compile(optimizer, loss=loss_fn)

print('--------------------------- MODEL LOADED -----------------------------')


def read_audio_file(file_path):
    with open(file_path, "rb") as f:
        audio_wave, sample_rate = sf.read(f)
    if sample_rate != REQUIRED_SAMPLE_RATE:
        raise ValueError(
            f"sample rate (={sample_rate}) of your files must be {REQUIRED_SAMPLE_RATE}"
        )
    audio_wave = audio_wave[:AUDIO_MAX_LEN]
    return audio_wave

def extract_face_emb_from_speech_per_row(row):
    print(row['id'])

    audio_wave = read_audio_file(row['path_var_len_audio'])
    audio_wave = tf.constant(audio_wave, dtype=tf.float64)
    audio_wave = tf.expand_dims(audio_wave, 0)
    #print('audio_wave')
    #print(audio_wave.shape)
    face_emb = model.predict(audio_wave)
    face_emb = face_emb.squeeze()
    #print(face_emb)
    embeddingsPickle = pickle.dumps(face_emb)

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



