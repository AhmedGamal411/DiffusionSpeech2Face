
import pickle

import os
import configparser
import torch
import numpy as np
from transformers import AutoProcessor, ASTModel
from scipy.io import wavfile
import torch
import pickle
from PIL import Image,ImageFilter
import random
from transformers import ViTImageProcessor, ViTModel


def extract_vision_transformer(q,output_folder,image_guide_path,blur_or_pixelate,image_size,boxBlurMin,
    boxBlurMax,gaussianBlurMin,gaussianBlurMax,pix_to_min,pix_to_max):

    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
    model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')

    im = Image.open(image_guide_path).convert('RGB')
    im = im.resize((image_size,image_size), resample=Image.Resampling.BILINEAR)

    blurOption = random.randint(1, 6)

    
    if(blurOption==1):
        imBlurred = im.filter(ImageFilter.BoxBlur(random.randint(boxBlurMin, boxBlurMax))) # 4 to 14
    elif(blurOption==2):
        imBlurred = im.filter(ImageFilter.GaussianBlur(random.randint(gaussianBlurMin, gaussianBlurMax))) # 4 to 8
    elif(blurOption==3):
        imBlurred = im.filter(ImageFilter.BoxBlur(random.randint(int(boxBlurMin/2),int(boxBlurMax/2)))).filter(
            ImageFilter.GaussianBlur(random.randint(int(gaussianBlurMin/2),int(gaussianBlurMax/2)))) 
    elif(blurOption==4 or blurOption==5 or blurOption==6):
        pix_to = random.randint(int(pix_to_min), int(pix_to_max))
        imBlurred = im.resize((pix_to,pix_to), resample=Image.Resampling.BILINEAR) # 8 to 24
        imBlurred = imBlurred.resize(im.size, Image.Resampling.NEAREST)
        #print('pixalated')

    if(blur_or_pixelate == 0):
        imBlurred = im


        #directory = "extractVggBlurred"
        #picFile = directory + "/pic.png"
        #if not os.path.exists(directory):
        #    os.makedirs(directory)

        #imBlurred.save(picFile)

        #img_path = picFile
    inputs = processor(images=imBlurred, return_tensors="pt")
    outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state
    features = last_hidden_states.cpu().detach().numpy()
        #print(last_hidden_states.cpu().detach().numpy().min()) #(1,197,768) max=1.23 min=-1.23
        
        #img = image.load_img(img_path, target_size=(224, 224))
        #x = image.img_to_array(img)
        #x = np.expand_dims(x, axis=0)
        #x = preprocess_input(x)

        #features = model.predict(x) #(1,7,7,512)
        #features = features.reshape(-1, features.shape[-1]) #(49,512)


    im.close()
    imBlurred.close()
    #mg.close()
    #os.remove(picFile)
    #print(features.shape) #(49, 512)
    #print("max:" + str(features.max())) #max is 72
    #print("min:" + str(features.min())) #min is 0
    embeddingsPickle = pickle.dumps(features)



    


    with open(output_folder + '/' + 'image_features.pickle', 'wb') as handle:
        pickle.dump(embeddingsPickle, handle, protocol=pickle.HIGHEST_PROTOCOL)
