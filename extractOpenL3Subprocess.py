
import openl3


def extractOpenL3Subprocess(audio,sr):
    model = openl3.models.load_audio_embedding_model(input_repr="mel128", content_type="music",
                                                 embedding_size=512)
    
    emb, ts = openl3.get_audio_embedding(audio, sr,hop_size=24/50,verbose=0,model=model)
    #queue.put(emb)
    print('Extracted OpenL3 features')
    return emb