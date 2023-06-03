import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["AUTOGRAPH_VERBOSITY"] = "0"
#TODO document jupyter
import pickle
import configparser
import sqlite3 as sl
import pandas as pd
import numpy as np
from PIL import Image
import logging


from textwrap import wrap
import os
import random


import keras_cv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.experimental.numpy as tnp
from keras_cv.models.stable_diffusion.clip_tokenizer import SimpleTokenizer
from keras_cv.models.stable_diffusion.diffusion_model import DiffusionModel
from keras_cv.models.stable_diffusion.image_encoder import ImageEncoder
from keras_cv.models.stable_diffusion.noise_scheduler import NoiseScheduler
from keras_cv.models.stable_diffusion.text_encoder import TextEncoder
from tensorflow import keras

import multiprocessing
from multiprocessing import Process
from threading import Thread

def trainBatch(epoch,offset,return_var):

    tf.get_logger().setLevel(logging.ERROR)
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


    configParser = configparser.RawConfigParser()   
    configFilePath = r'configuration.txt'
    configParser.read(configFilePath)
    datasetPathDatabase =  configParser.get('COMMON', 'datasetPathDatabase') + '/dataset.db'
    db_chunk = int(configParser.get('fineTuneStableDiffusionTraining', 'db_chunk'))
    dev_mode = int(configParser.get('fineTuneStableDiffusionTraining', 'dev_mode'))

    #continue_from_epoch = int(configParser.get('fineTuneStableDiffusionTraining', 'continue_from_epoch'))
    #continue_from_offset = int(configParser.get('fineTuneStableDiffusionTraining', 'continue_from_offset'))
    #continue_from_epoch_and_offset = int(configParser.get('fineTuneStableDiffusionTraining', 'continue_from_epoch_and_offset'))

    unconditional_guidance_scale = int(configParser.get('fineTuneStableDiffusionTraining', 'unconditional_guidance_scale'))

    con = sl.connect(datasetPathDatabase)

    def speaker_emb_preprocess(speaker_emb2):
        '''
        A function to take a texual promt and convert it into embeddings
        '''
        #if maxlen is None: maxlen = tokenizer.model_max_length
        #inp = tokenizer(prompts, padding="max_length", max_length=maxlen, truncation=True, return_tensors="pt") 
        #return text_encoder(inp.input_ids.to("cuda"))[0].half()

        #speaker_emb2 = speaker_emb2.squeeze()
        speaker_emb2 = pickle.loads(speaker_emb2)
        #print(speaker_emb2.shape)
        speaker_emb2 = speaker_emb2.squeeze()
        speaker_emb2 = np.pad(speaker_emb2, (288), 'constant', constant_values=(0))
        #print(speaker_emb2.shape)
        speaker_emb2 = np.tile(speaker_emb2, (1, 1))

        speaker_emb2 = np.array(speaker_emb2).tolist()
        
        
        #speaker_emb2 = torch.from_numpy(speaker_emb2).type(torch.FloatTensor)
        
        #return speaker_emb2.to("cuda").half()
        return speaker_emb2


    def getImage(face_path):
        im = Image.open(face_path)
        im.load() # required for png.split()

        im2 = Image.new("RGB", im.size, (255, 255, 255))
        im2.paste(im, mask=im.split()[3]) # 3 is the alpha channel
        im3 = np.array(im2)
        im4 = np.rollaxis(im3,2)
        #im4 = torch.from_numpy(im4).type(torch.FloatTensor)
        
        #return im4.to("cuda").half()
        return im4

    # The padding token and maximum prompt length are specific to the text encoder.
    # If you're using a different text encoder be sure to change them accordingly.
    PADDING_TOKEN = 49407
    MAX_PROMPT_LENGTH = 77

    # Load the tokenizer.
    tokenizer = SimpleTokenizer()

    #  Method to tokenize and pad the tokens.
    def process_text(caption):
        tokens = tokenizer.encode(caption)
        tokens = tokens + [PADDING_TOKEN] * (MAX_PROMPT_LENGTH - len(tokens))
        return np.array(tokens)


    RESOLUTION = 128
    AUTO = tf.data.AUTOTUNE
    POS_IDS = tf.convert_to_tensor([list(range(MAX_PROMPT_LENGTH))], dtype=tf.int32)

    augmenter = keras_cv.layers.Augmenter(
        layers=[
            keras_cv.layers.CenterCrop(RESOLUTION, RESOLUTION),
            keras_cv.layers.RandomFlip(),
            tf.keras.layers.Rescaling(scale=1.0 / 127.5, offset=-1),
        ]
    )
    text_encoder = TextEncoder(MAX_PROMPT_LENGTH)


    def process_image(image_path, tokenized_text,speaker_emb):
        #y = tf.py_function(func=show, inp=[speaker_emb], Tout=tf.float32)


        image = tf.io.read_file(image_path)
        image = tf.io.decode_png(image, 3)
        image = tf.image.resize(image, (RESOLUTION, RESOLUTION))
        return image, tokenized_text,speaker_emb


    def apply_augmentation(image_batch, token_batch,speaker_emb):
        return augmenter(image_batch), token_batch,speaker_emb


    def run_text_encoder(image_batch, token_batch,speaker_emb):

        speaker_emb = tf.cast(speaker_emb, tf.float32)
        
        textEncoderOp = text_encoder([token_batch, POS_IDS], training=False)

        #print(textEncoderOp.shape)
        textEncoderOp = textEncoderOp[:,:-1,:]

        #print(speaker_emb.shape)
        #print(textEncoderOp.shape)

        textEncoderOp = tf.concat([textEncoderOp, speaker_emb], 1)

        #print(textEncoderOp.shape)


        
        
        return (
            image_batch,
            token_batch,
            speaker_emb,
            textEncoderOp,
        )


    def prepare_dict(image_batch, token_batch, speaker_emb,encoded_text_batch):
        return {
            "images": image_batch,
            "tokens": token_batch,
            "index":speaker_emb,
            "encoded_text": encoded_text_batch,
        }


    def prepare_dataset(image_paths, tokenized_texts,speaker_emb , batch_size=1):
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, tokenized_texts, speaker_emb))
        dataset = dataset.shuffle(batch_size * 10)
        dataset = dataset.map(process_image, num_parallel_calls=AUTO).batch(batch_size)
        dataset = dataset.map(apply_augmentation, num_parallel_calls=AUTO)
        dataset = dataset.map(run_text_encoder, num_parallel_calls=AUTO)
        dataset = dataset.map(prepare_dict, num_parallel_calls=AUTO)
        return dataset.prefetch(AUTO)


    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', -1)
    pd.options.mode.chained_assignment = None


    class Trainer(tf.keras.Model):
        # Reference:
        # https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py

        def __init__(
            self,
            diffusion_model,
            vae,
            noise_scheduler,
            use_mixed_precision=False,
            max_grad_norm=1.0,
            **kwargs
        ):
            super().__init__(**kwargs)

            self.diffusion_model = diffusion_model
            self.vae = vae
            self.noise_scheduler = noise_scheduler
            self.max_grad_norm = max_grad_norm

            self.use_mixed_precision = use_mixed_precision
            self.vae.trainable = False

        def train_step(self, inputs):
            images = inputs["images"]
            encoded_text = inputs["encoded_text"]
            batch_size = tf.shape(images)[0]

            with tf.GradientTape() as tape:
                # Project image into the latent space and sample from it.
                latents = self.sample_from_encoder_outputs(self.vae(images, training=False))
                # Know more about the magic number here:
                # https://keras.io/examples/generative/fine_tune_via_textual_inversion/
                latents = latents * 0.18215

                # Sample noise that we'll add to the latents.
                noise = tf.random.normal(tf.shape(latents))

                # Sample a random timestep for each image.
                timesteps = tnp.random.randint(
                    0, self.noise_scheduler.train_timesteps, (batch_size,)
                )

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process).
                noisy_latents = self.noise_scheduler.add_noise(
                    tf.cast(latents, noise.dtype), noise, timesteps
                )

                # Get the target for loss depending on the prediction type
                # just the sampled noise for now.
                target = noise  # noise_schedule.predict_epsilon == True

                # Predict the noise residual and compute loss.
                timestep_embedding = tf.map_fn(
                    lambda t: self.get_timestep_embedding(t), timesteps, dtype=tf.float32
                )
                timestep_embedding = tf.squeeze(timestep_embedding, 1)
                model_pred = self.diffusion_model(
                    [noisy_latents, timestep_embedding, encoded_text], training=True
                )
                loss = self.compiled_loss(target, model_pred)
                if self.use_mixed_precision:
                    loss = self.optimizer.get_scaled_loss(loss)

            # Update parameters of the diffusion model.
            trainable_vars = self.diffusion_model.trainable_variables
            gradients = tape.gradient(loss, trainable_vars)
            if self.use_mixed_precision:
                gradients = self.optimizer.get_unscaled_gradients(gradients)
            gradients = [tf.clip_by_norm(g, self.max_grad_norm) for g in gradients]
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))

            return {m.name: m.result() for m in self.metrics}

        def get_timestep_embedding(self, timestep, dim=320, max_period=10000):
            half = dim // 2
            log_max_preiod = tf.math.log(tf.cast(max_period, tf.float32))
            freqs = tf.math.exp(
                -log_max_preiod * tf.range(0, half, dtype=tf.float32) / half
            )
            args = tf.convert_to_tensor([timestep], dtype=tf.float32) * freqs
            embedding = tf.concat([tf.math.cos(args), tf.math.sin(args)], 0)
            embedding = tf.reshape(embedding, [1, -1])
            return embedding

        def sample_from_encoder_outputs(self, outputs):
            mean, logvar = tf.split(outputs, 2, axis=-1)
            logvar = tf.clip_by_value(logvar, -30.0, 20.0)
            std = tf.exp(0.5 * logvar)
            sample = tf.random.normal(tf.shape(mean), dtype=mean.dtype)
            return mean + std * sample

        def save_weights(self, filepath, overwrite=True, save_format=None, options=None):
            # Overriding this method will allow us to use the `ModelCheckpoint`
            # callback directly with this trainer class. In this case, it will
            # only checkpoint the `diffusion_model` since that's what we're training
            # during fine-tuning.
            self.diffusion_model.save_weights(
                filepath=filepath,
                overwrite=overwrite,
                save_format=save_format,
                options=options,
            )


    # Enable mixed-precision training if the underlying GPU has tensor cores.
    USE_MP = False
    if USE_MP:
        keras.mixed_precision.set_global_policy("mixed_float16")

    image_encoder = ImageEncoder(RESOLUTION, RESOLUTION)
    diffusion_ft_trainer = Trainer(
        diffusion_model=DiffusionModel(RESOLUTION, RESOLUTION, MAX_PROMPT_LENGTH),
        # Remove the top layer from the encoder, which cuts off the variance and only
        # returns the mean.
        vae=tf.keras.Model(
            image_encoder.input,
            image_encoder.layers[-2].output,
        ),
        noise_scheduler=NoiseScheduler(),
        use_mixed_precision=USE_MP,
    )

    # These hyperparameters come from this tutorial by Hugging Face:
    # https://huggingface.co/docs/diffusers/training/text2image
    lr = 1e-5
    beta_1, beta_2 = 0.9, 0.999
    weight_decay = (1e-2,)
    epsilon = 1e-08

    optimizer = tf.keras.optimizers.experimental.AdamW(
        learning_rate=lr,
        weight_decay=weight_decay,
        beta_1=beta_1,
        beta_2=beta_2,
        epsilon=epsilon,
    )
    diffusion_ft_trainer.compile(optimizer=optimizer, loss="mse")


    ckpt_path = "finetuned_stable_diffusion.h5"
    ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
        ckpt_path,
        save_weights_only=True,
        monitor="loss",
        mode="min",
    )



    con = sl.connect(datasetPathDatabase)
    data = con.execute("SELECT V.ID, V.VIDEO_PATH, V.AGE, " + 
                    "'This person is '|| V.ETHNICITY || '. ' CAPTION_E, " +
                    "'This person is a '|| lower(V.GENDER) || '. ' CAPTION_G, " +
                    "A.SPEAKER_EMB, "+
                    "'This person speaks ' || A.LANG || '. ' CAPTION_L, "+
                    "F.FACE_PATH, "+
                    "'The face of a person. ' CAPTION  FROM VIDEO V "+
                    "INNER JOIN AUDIO A ON V.ID = A.VIDEO_ID INNER JOIN FACE F ON V.ID = F.VIDEO_ID " +
                    "LIMIT "+ str(db_chunk) +" OFFSET " + str(offset))
    dataGotten = data.fetchall()
    
    #print('gotten ' + str(db_chunk))

    if(len(dataGotten) == 0):
        con.close()
        return_var = 1
        return

    
    df = pd.DataFrame(dataGotten,columns = ['ID','VIDEO_PATH','AGE','caption_e','caption_g','SPEAKER_EMB','caption_l','image_path','caption'])
    df3 = df[["image_path","caption","caption_e","caption_g","caption_l"]]
    df3 = df3.fillna('')

    df3['caption_e'] = df3['caption_e'].apply(lambda x: x if random.random() < 0.2 else '')
    df3['caption_g'] = df3['caption_g'].apply(lambda x: x if random.random() < 0.2 else '')
    df3['caption_l'] = df3['caption_l'].apply(lambda x: x if random.random() < 0.2 else '')

    df3['caption'] = df3['caption'] + df3['caption_e'] + df3['caption_g'] + df3['caption_l']

    df3 = df3[["image_path","caption"]]


    data_frame = df3
    #print(df3)

    # Collate the tokenized captions into an array.
    tokenized_texts = np.empty((len(data_frame), MAX_PROMPT_LENGTH))

    all_captions = list(data_frame["caption"].values)
    for i, caption in enumerate(all_captions):
        tokenized_texts[i] = process_text(caption)

    data_frame['SPEAKER_EMB'] = df['SPEAKER_EMB']

    

    for index, row in data_frame.iterrows():
        x = speaker_emb_preprocess(data_frame.loc[index,"SPEAKER_EMB"])
        x = [x]
        data_frame.loc[index,"SPEAKER_EMB"] = x
    
    data_frame_length = len(data_frame)
    a = np.zeros(shape=(data_frame_length, 768))
    for index,row in data_frame.iterrows():
        a[index] = ( np.asarray(row[2], dtype=np.float32).squeeze())
        #print(row['image_path'])
    a.squeeze()
    a = np.expand_dims(a, axis=1)

    emb = np.expand_dims(a[0], axis=1)
    ##encoded_text = np.concatenate((encoded_text, emb), axis=1)


    # Prepare the dataset.
    training_dataset = prepare_dataset(
        np.array(data_frame["image_path"]), tokenized_texts, a,batch_size=1
    )

    # Take a sample batch and investigate.
    #sample_batch = next(iter(training_dataset))

    #for k in sample_batch:
    #    print(k, sample_batch[k].shape)

    #plt.figure(figsize=(20, 10))


    #ax = plt.subplot(1, 4, 1)
    #plt.imshow((sample_batch["images"][0] + 1) / 2)

    #text = tokenizer.decode(sample_batch["tokens"][0].numpy().squeeze())
    #text = text.replace("<|startoftext|>", "")
    #text = text.replace("<|endoftext|>", "")
    #text = "\n".join(wrap(text, 12))
    #plt.title(text, fontsize=15)

    #plt.axis("off")

    if(dev_mode == 0):

        weights_path = "finetuned_stable_diffusion.h5"

        diffusion_ft_trainer.diffusion_model.load_weights(weights_path)


        print("Live Mode: Training: epoch:" + str(epoch) + " offset:" + str(offset))

        diffusion_ft_trainer.fit(training_dataset, epochs=1, callbacks=[ckpt_callback])

        diffusion_ft_trainer.diffusion_model.save_weights(weights_path)

        

        weights_path = "finetuned_stable_diffusion.h5"
        img_height = img_width = 128
        s2f_model = keras_cv.models.StableDiffusion(
            img_width=img_width, img_height=img_height
        )
        # We just reload the weights of the fine-tuned diffusion model.
        s2f_model.diffusion_model.load_weights(weights_path)

        testsFolder = 'tests'
        if not os.path.exists(testsFolder):
            os.makedirs(testsFolder)
        prompt = "The face of a person. "
        images_to_generate = 1

        outputs = {}
        encoded_text = s2f_model.encode_text(prompt)
        encoded_text = encoded_text[:,:-1,:]

        emb = np.expand_dims(a[0], axis=1)
        encoded_text = np.concatenate((encoded_text, emb), axis=1)

        generated_images = s2f_model.generate_image(
            encoded_text = encoded_text, batch_size=images_to_generate, 
            unconditional_guidance_scale=unconditional_guidance_scale
        )
        outputs.update({prompt: generated_images})

        images = outputs[prompt]
        for prompt in outputs:
            for i in range(len(images)):
                image_to_be_saved = Image.fromarray(images[i])
                image_to_be_saved.save(testsFolder + '/' + 'epoch' + str(epoch) + '-offset' + str(offset) + 
                                    '-image' + str(i) + '.jpg')
                



    else:
        print("Dev Mode: Training: epoch:" + str(epoch) + " offset:" + str(offset))
        
    con.close()
    return_var = 0
    return