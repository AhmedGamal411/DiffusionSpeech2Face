
import openl3


def extractOpenL3Subprocess(audio,hop_size,sr):
    model = openl3.models.load_audio_embedding_model(input_repr="mel128", content_type="music",
                                                 embedding_size=512)
    
    emb, ts = openl3.get_audio_embedding(audio, sr,hop_size=hop_size,verbose=0,model=model)
    #queue.put(emb)
    #print(emb.shape)
    print('Extracted OpenL3 features')
    return emb