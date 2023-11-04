import numpy as np
from PIL import Image
import torch
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter
import pandas as pd


def train_batch_unet1(input0,input2,output,model_filename,inner_epochs,batch_size,sample_every,save_model_every,image_size,unet1_dim,unet2_dim,timesteps,begin_with_image_size,unet1_image_size,imagen_samples,sample_probability,dask_chunk,ignore_image_guide):
    from torch.utils.data import TensorDataset, DataLoader
    import time
    UNET = 1
    print("Training Unet No. 1")
    


    input0 = torch.from_numpy(input0)
    input0 = input0.to(torch.float)


    input2 = torch.from_numpy(input2)
    input2 = input2.to(torch.float)

    output = torch.from_numpy(output)
    output = output.to(torch.float)

    input0 = input0.squeeze()
    input2 = input2.squeeze()
    output = output.squeeze()




    my_dataset = TensorDataset(output,input0) # create your datset
    
    my_dataloader = DataLoader(my_dataset) # create your dataloader
    



    try:
        from imagen_pytorch import Unet, Imagen, ImagenTrainer
        from imagen_pytorch.data import Dataset
        from imagen_pytorch.t5 import t5_encode_text
        t5_encode_text(texts="test").max()
    except:
        from imagen_pytorch import Unet, Imagen, ImagenTrainer
        from imagen_pytorch.data import Dataset
        from imagen_pytorch.t5 import t5_encode_text
        t5_encode_text(texts="test").max()

    ground_truth = output[0].cpu().detach().numpy()
    

    ground_truth = np.moveaxis(ground_truth, 0, -1)
    

    ground_truth = ground_truth * 255
    ground_truth = ground_truth.astype(np.uint8)

    ground_truth = Image.fromarray(ground_truth)
    


    from imagen_pytorch import Unet, Imagen, ImagenTrainer,NullUnet
    from imagen_pytorch.data import Dataset

    # unets for unconditional imagen


    unet0 = NullUnet()  # add a placeholder "null" unet for the base unet

    unet1 = Unet(
        dim = unet1_dim,
        cond_dim = 512,
        dim_mults = (1, 2, 4, 8),
        num_resnet_blocks = 3,
        layer_attns = (False, True, True, True),
        layer_cross_attns = (False, True, True, True)
    )

    unet2 = Unet(
        dim = unet2_dim,
        cond_dim = 512,
        dim_mults = (1, 2, 4, 8),
        num_resnet_blocks = (2, 4, 8, 8),
        layer_attns = (False, False, False, True),
        layer_cross_attns = (False, False, False, True)
    )

    if(ignore_image_guide):
        unets = (unet1, unet2)
        image_sizes= (unet1_image_size, image_size)
        unet_to_train = UNET #1
        start_image_or_video = None
    else:
        unets = (unet0,unet1, unet2)
        image_sizes= (begin_with_image_size,unet1_image_size, image_size)
        unet_to_train = UNET + 1 #2
        start_image_or_video = input2[:1,:]

    #print(input2[:1,:])
    #print(input2)

    imagen = Imagen(
        unets = unets,
        image_sizes = image_sizes,
        timesteps = timesteps,
        cond_drop_prob = 0.1
    ).cuda()
    


    trainer = ImagenTrainer(
        imagen = imagen,
        split_valid_from_train = False # whether to split the validation dataset from the training
    ).cuda()

    


    # instantiate your dataloader, which returns the necessary inputs to the DDPM as tuple in the order of images, text embeddings, then text masks. in this case, only images is returned as it is unconditional training

    

    trainer.add_train_dataset(my_dataset, batch_size = batch_size * 16)
    
    # working training loop

    import os
    import time

    if not os.path.exists(imagen_samples):
        os.makedirs(imagen_samples)

    now =time.time()
    seconds = now


    from pathlib import Path

    my_file = Path(model_filename)
    if my_file.is_file():
        print('Using model file ' + model_filename)
        trainer.load(model_filename)

    import math
    import random
    import pickle
    loss_list = []
    for i in range(inner_epochs):
        
        loss = trainer.train_step(unet_number = unet_to_train,max_batch_size = batch_size)
        if not (i % 10):
            print(f'loss: {loss}')
        loss_list.append(loss)

        #if not (i % 50):
        #    valid_loss = trainer.valid_step(unet_number = 2, max_batch_size =  batch_size)
        #    print(f'valid loss: {valid_loss}')

        if not (i % sample_every) and trainer.is_main and random.choices([True, False], weights=[sample_probability, 100-sample_probability])[0]: # is_main makes sure this can run in distributed
            cond_scale = random.uniform(5.1, 9.9)
            if not os.path.exists(str(seconds)):
                os.makedirs(imagen_samples + "/" + str(seconds))
            ground_truth.save(imagen_samples + '/' + str(seconds) + '/ground_truth.png')
            
            images = trainer.sample(text_embeds=input0[:1, :],start_image_or_video = start_image_or_video,start_at_unet_number = unet_to_train
                                    ,stop_at_unet_number=unet_to_train,batch_size = 1, return_pil_images = True,cond_scale=cond_scale) # returns List[Image]
            images[0].save(imagen_samples + '/' + str(seconds) + f'/sample-{i // 100}'+'-'+str(int(cond_scale))+'-'+'.png')

        #if not (i % save_model_every):
        #    trainer.save(model_filename)
    

    trainer.save(model_filename)
    

    with open('loss_list_1_temp.pickle', 'wb') as handle:
        pickle.dump(loss_list, handle)

    if(UNET == 1):
        with open('loss_list_1_temp.pickle', 'rb') as handle:
            loss_list = pickle.load(handle)
    else:
        with open('loss_list_2_temp.pickle', 'rb') as handle:
            loss_list = pickle.load(handle)
    
    if(UNET == 1):
        my_file = Path(model_filename + 'loss_total_1.picke')
        if my_file.is_file():
            with open(model_filename + 'loss_total_1.picke', 'rb') as handle:
                loss_total = pickle.load(handle)
        else:
            loss_total = []
    else:
        my_file = Path(model_filename + 'loss_total_2.picke')
        if my_file.is_file():
            with open(model_filename + 'loss_total_2.picke', 'rb') as handle:
                loss_total = pickle.load(handle)
        else:
            loss_total = []
    
    #print(loss_list)
    #print(loss_total)
    loss_total.extend(loss_list)

    if(UNET == 1):
        with open(model_filename +'loss_total_1.picke', 'wb') as handle:
            pickle.dump(loss_total, handle)
    else:
        with open(model_filename +'loss_total_2.picke', 'wb') as handle:
            pickle.dump(loss_total, handle)


    fig = plt.figure()
    plt.plot(loss_total)
    plt.title("Training Loss")
    plt.xlabel("Training Sample (~x" + str(int(dask_chunk)) + ")")
    plt.ylabel("MSE Loss")
    



    if(UNET == 1):
        if(os.path.isfile(model_filename + 'loss_1_plot.png')):
            os.remove(model_filename + 'loss_1_plot.png')
        fig.savefig(model_filename + 'loss_1_plot.png')
    else:
        if(os.path.isfile(model_filename + 'loss_2_plot.png')):
            os.remove(model_filename + 'loss_2_plot.png')
        fig.savefig(model_filename + 'loss_2_plot.png')
    plt.close()

    try:
        fig = plt.figure()
        plt.plot(np.arange(len(loss_total[1000::])) + 1000,loss_total[1000::],'.')

        plt.axvline(x=1000,linestyle='--',color='green',label='1000 inner epochs')
        plt.axvline(x=17000,linestyle='--',color='purple',label='100 inner epochs - 100000 unique samples seen' )
        plt.axvline(x=23100,linestyle='-.',color='black',label='1 inner epoch - end of 1st epoch ')
        plt.legend(bbox_to_anchor = (1.0, 1), loc = 'upper right')

        yhat = savgol_filter(loss_total, 1000, 3)
        plt.plot(np.arange(len(loss_total[1000::])) + 1000,yhat[1000::],'r')
        plt.title("Training Loss")
        plt.xlabel("Training Sample (~x" + str(int(dask_chunk)) + ")")
        plt.ylabel("MSE Loss")
        



        if(UNET == 1):
            if(os.path.isfile(model_filename + 'loss_zoomed_1_plot.png')):
                os.remove(model_filename + 'loss_zoomed_1_plot.png')
            fig.savefig(model_filename + 'loss_zoomed_1_plot.png')
        else:
            if(os.path.isfile(model_filename + 'loss_zoomed_2_plot.png')):
                os.remove(model_filename + 'loss_zoomed_2_plot.png')
            fig.savefig(model_filename + 'loss_zoomed_2_plot.png')
        plt.close()


        fig = plt.figure()
        smoothed = yhat[1000::]
        smoothed = savgol_filter(smoothed, 500, 1)
        price_series = pd.Series(smoothed)
        price_series = price_series.pct_change().to_numpy()
        plt.plot(np.arange(len(loss_total[1000::])) + 1000,price_series)
        plt.axhline(linestyle='--',color='red')
        plt.axvline(x=1000,linestyle='--',color='green',label='1000 inner epochs')
        plt.axvline(x=17000,linestyle='--',color='purple',label='100 inner epochs - 100000 unique samples seen' )
        plt.axvline(x=23100,linestyle='-.',color='black',label='1 inner epoch - end of 1st epoch ')
        plt.legend(loc = 'lower center')
        plt.title("Smoothed Rate of Change of Training Loss")
        plt.xlabel("Training Sample (~x" + str(int(dask_chunk)) + ")")
        plt.ylabel("Rate of Change")
        #plt.show()

        if(UNET == 1):
            if(os.path.isfile(model_filename + 'loss_roc_1_plot.png')):
                os.remove(model_filename + 'loss_roc_1_plot.png')
            fig.savefig(model_filename + 'loss_roc_1_plot.png')
        else:
            if(os.path.isfile(model_filename + 'loss_roc_2_plot.png')):
                os.remove(model_filename + 'loss_roc_2_plot.png')
            fig.savefig(model_filename + 'loss_roc_2_plot.png')
        plt.close()
    except:
        pass



def train_batch_unet2(input0,input2,output,model_filename,inner_epochs,batch_size,sample_every,save_model_every,image_size,unet1_dim,unet2_dim,timesteps,begin_with_image_size,unet1_image_size,imagen_samples,sample_probability,dask_chunk,ignore_image_guide):
    from torch.utils.data import TensorDataset, DataLoader

    print("Training Unet No. 2")
    UNET = 2

    
    input0 = torch.from_numpy(input0)
    input0 = input0.to(torch.float)

    input2 = torch.from_numpy(input2)
    input2 = input2.to(torch.float)

    output = torch.from_numpy(output)
    output = output.to(torch.float)

    input0 = input0.squeeze()
    input2 = input2.squeeze()
    output = output.squeeze()



    my_dataset = TensorDataset(output,input0) # create your datset
    my_dataloader = DataLoader(my_dataset) # create your dataloader



    try:
        from imagen_pytorch import Unet, Imagen, ImagenTrainer
        from imagen_pytorch.data import Dataset
        from imagen_pytorch.t5 import t5_encode_text
        t5_encode_text(texts="test").max()
    except:
        from imagen_pytorch import Unet, Imagen, ImagenTrainer
        from imagen_pytorch.data import Dataset
        from imagen_pytorch.t5 import t5_encode_text
        t5_encode_text(texts="test").max()

    ground_truth = output[0].cpu().detach().numpy()
    ground_truth = np.moveaxis(ground_truth, 0, -1)
    ground_truth = ground_truth * 255
    ground_truth = ground_truth.astype(np.uint8)
    ground_truth = Image.fromarray(ground_truth)

    from imagen_pytorch import Unet, Imagen, ImagenTrainer,NullUnet
    from imagen_pytorch.data import Dataset

    # unets for unconditional imagen


    unet0 = NullUnet()  # add a placeholder "null" unet for the base unet

    unet1 = Unet(
        dim = unet1_dim,
        cond_dim = 512,
        dim_mults = (1, 2, 4, 8),
        num_resnet_blocks = 3,
        layer_attns = (False, True, True, True),
        layer_cross_attns = (False, True, True, True)
    )

    unet2 = Unet(
        dim = unet2_dim,
        cond_dim = 512,
        dim_mults = (1, 2, 4, 8),
        num_resnet_blocks = (2, 4, 8, 8),
        layer_attns = (False, False, False, True),
        layer_cross_attns = (False, False, False, True)
    )

    if(ignore_image_guide):
        unets = (unet1, unet2)
        image_sizes= (unet1_image_size, image_size)
        unet_to_train = UNET #2
        start_image_or_video = None
    else:
        unets = (unet0,unet1, unet2)
        image_sizes= (begin_with_image_size,unet1_image_size, image_size)
        unet_to_train = UNET + 1 #3
        start_image_or_video = input2[:1,:]

    #print(input2[:1,:])
    #print(input2)

    imagen = Imagen(
        unets = unets,
        image_sizes = image_sizes,
        timesteps = timesteps,
        cond_drop_prob = 0.1
    ).cuda()


    trainer = ImagenTrainer(
        imagen = imagen,
        split_valid_from_train = False # whether to split the validation dataset from the training
    ).cuda()

    # instantiate your dataloader, which returns the necessary inputs to the DDPM as tuple in the order of images, text embeddings, then text masks. in this case, only images is returned as it is unconditional training



    trainer.add_train_dataset(my_dataset, batch_size = batch_size * 16)

    # working training loop

    import os
    import time

    if not os.path.exists(imagen_samples):
        os.makedirs(imagen_samples)

    now =time.time()
    seconds = now

    from pathlib import Path

    my_file = Path(model_filename)
    if my_file.is_file():
        print('Using model file ' + model_filename)
        trainer.load(model_filename)

    import math
    import random
    import pickle
    loss_list = []
    for i in range(inner_epochs):
        loss = trainer.train_step(unet_number = unet_to_train,max_batch_size = batch_size)
        if not (i % 10):
            print(f'loss: {loss}')
        loss_list.append(loss)

        #if not (i % 50):
        #    valid_loss = trainer.valid_step(unet_number = 3, max_batch_size =  batch_size)
        #    print(f'valid loss: {valid_loss}')

        if not (i % sample_every) and trainer.is_main and random.choices([True, False], weights=[sample_probability, 100-sample_probability])[0]: # is_main makes sure this can run in distributed
            cond_scale = random.uniform(5.1, 9.9)
            if not os.path.exists(str(seconds)):
                os.makedirs(imagen_samples + "/" + str(seconds))

            ground_truth.save(imagen_samples + '/' + str(seconds) + '/ground_truth.png')

            images = trainer.sample(text_embeds=input0[:1, :],start_image_or_video = start_image_or_video,start_at_unet_number = unet_to_train - 1
                                    ,stop_at_unet_number=unet_to_train,batch_size = 1, return_pil_images = True,cond_scale=cond_scale) # returns List[Image]
            images[0].save(imagen_samples + '/' + str(seconds) + f'/sample-{i // 100}'+'-'+str(int(cond_scale))+'-'+'.png')

        #if not (i % save_model_every):
        #    trainer.save(model_filename)

    trainer.save(model_filename)
    with open('loss_list_2_temp.pickle', 'wb') as handle:
        pickle.dump(loss_list, handle)

    if(UNET == 1):
        with open('loss_list_1_temp.pickle', 'rb') as handle:
            loss_list = pickle.load(handle)
    else:
        with open('loss_list_2_temp.pickle', 'rb') as handle:
            loss_list = pickle.load(handle)
    
    if(UNET == 1):
        my_file = Path(model_filename + 'loss_total_1.picke')
        if my_file.is_file():
            with open(model_filename + 'loss_total_1.picke', 'rb') as handle:
                loss_total = pickle.load(handle)
        else:
            loss_total = []
    else:
        my_file = Path(model_filename + 'loss_total_2.picke')
        if my_file.is_file():
            with open(model_filename + 'loss_total_2.picke', 'rb') as handle:
                loss_total = pickle.load(handle)
        else:
            loss_total = []
    
    #print(loss_list)
    #print(loss_total)
    loss_total.extend(loss_list)

    if(UNET == 1):
        with open(model_filename +'loss_total_1.picke', 'wb') as handle:
            pickle.dump(loss_total, handle)
    else:
        with open(model_filename +'loss_total_2.picke', 'wb') as handle:
            pickle.dump(loss_total, handle)


    fig = plt.figure()
    plt.plot(loss_total)
    plt.title("Training Loss")
    plt.xlabel("Training Sample (~x" + str(int(dask_chunk)) + ")")
    plt.ylabel("MSE Loss")
    



    if(UNET == 1):
        if(os.path.isfile(model_filename + 'loss_1_plot.png')):
            os.remove(model_filename + 'loss_1_plot.png')
        fig.savefig(model_filename + 'loss_1_plot.png')
    else:
        if(os.path.isfile(model_filename + 'loss_2_plot.png')):
            os.remove(model_filename + 'loss_2_plot.png')
        fig.savefig(model_filename + 'loss_2_plot.png')
    plt.close()


    try:

        fig = plt.figure()
        plt.plot(np.arange(len(loss_total[2000::])) + 2000,loss_total[2000::],'.')

        plt.axvline(x=2000,linestyle='--',color='green',label='500 inner epochs')
        plt.axvline(x=10000,linestyle='--',color='purple',label='100 inner epochs')
        plt.axvline(x=13500,linestyle='-.',color='black',label='1 inner epoch')
        plt.legend(bbox_to_anchor = (1.0, 1), loc = 'upper right')

        yhat = savgol_filter(loss_total, 1000, 3)
        plt.plot(np.arange(len(loss_total[2000::])) + 2000,yhat[2000::],'r')
        plt.title("Training Loss")
        plt.xlabel("Training Sample (~x" + str(int(dask_chunk)) + ")")
        plt.ylabel("MSE Loss")
        



        if(UNET == 1):
            if(os.path.isfile(model_filename + 'loss_zoomed_1_plot.png')):
                os.remove(model_filename + 'loss_zoomed_1_plot.png')
            fig.savefig(model_filename + 'loss_zoomed_1_plot.png')
        else:
            if(os.path.isfile(model_filename + 'loss_zoomed_2_plot.png')):
                os.remove(model_filename + 'loss_zoomed_2_plot.png')
            fig.savefig(model_filename + 'loss_zoomed_2_plot.png')
        plt.close()


        fig = plt.figure()
        smoothed = yhat[2000::]
        smoothed = savgol_filter(smoothed, 500, 1)
        price_series = pd.Series(smoothed)
        price_series = price_series.pct_change().to_numpy()
        plt.plot(np.arange(len(loss_total[2000::])) + 2000,price_series)
        plt.axhline(linestyle='--',color='red')
        plt.axvline(x=2000,linestyle='--',color='green',label='500 inner epochs')
        plt.axvline(x=10000,linestyle='--',color='purple',label='100 inner epochs')
        plt.axvline(x=13500,linestyle='-.',color='black',label='1 inner epoch')
        plt.legend(bbox_to_anchor = (1.0, 1), loc = 'upper right')
        plt.legend(loc = 'lower center')
        plt.title("Smoothed Rate of Change of Training Loss")
        plt.xlabel("Training Sample (~x" + str(int(dask_chunk)) + ")")
        plt.ylabel("Rate of Change")
        #plt.show()

        if(UNET == 1):
            if(os.path.isfile(model_filename + 'loss_roc_1_plot.png')):
                os.remove(model_filename + 'loss_roc_1_plot.png')
            fig.savefig(model_filename + 'loss_roc_1_plot.png')
        else:
            if(os.path.isfile(model_filename + 'loss_roc_2_plot.png')):
                os.remove(model_filename + 'loss_roc_2_plot.png')
            fig.savefig(model_filename + 'loss_roc_2_plot.png')
        plt.close()
    except:
        pass



def sample_batch_unet(input0,input2,model_filename,image_size,unet1_dim,unet2_dim,timesteps,begin_with_image_size,unet1_image_size,cond_scale,chunk,ignore_image_guide):
    from torch.utils.data import TensorDataset, DataLoader
    import time
    import gc
    import dask.array as da

    try:
        from imagen_pytorch import Unet, Imagen, ImagenTrainer
        from imagen_pytorch.data import Dataset
        from imagen_pytorch.t5 import t5_encode_text
        t5_encode_text(texts="test").max()
    except:
        from imagen_pytorch import Unet, Imagen, ImagenTrainer
        from imagen_pytorch.data import Dataset
        from imagen_pytorch.t5 import t5_encode_text
        t5_encode_text(texts="test").max()



    from imagen_pytorch import Unet, Imagen, ImagenTrainer,NullUnet
    from imagen_pytorch.data import Dataset

    # unets for unconditional imagen


    unet0 = NullUnet()  # add a placeholder "null" unet for the base unet

    unet1 = Unet(
        dim = unet1_dim,
        cond_dim = 512,
        dim_mults = (1, 2, 4, 8),
        num_resnet_blocks = 3,
        layer_attns = (False, True, True, True),
        layer_cross_attns = (False, True, True, True)
    )

    unet2 = Unet(
        dim = unet2_dim,
        cond_dim = 512,
        dim_mults = (1, 2, 4, 8),
        num_resnet_blocks = (2, 4, 8, 8),
        layer_attns = (False, False, False, True),
        layer_cross_attns = (False, False, False, True)
    )

    if(ignore_image_guide):
        unets = (unet1, unet2)
        image_sizes= (unet1_image_size, image_size)
        unet_to_stop_sample= 2
    else:
        unets = (unet0,unet1, unet2)
        image_sizes= (begin_with_image_size,unet1_image_size, image_size)
        unet_to_stop_sample = 3
        

    #print(input2[:1,:])
    #print(input2)

    imagen = Imagen(
        unets = unets,
        image_sizes = image_sizes,
        timesteps = timesteps,
        cond_drop_prob = 0.1
    ).cuda()

    trainer = ImagenTrainer(imagen)
    


    #trainer = ImagenTrainer(
    #    imagen = imagen,
    #    split_valid_from_train = False # whether to split the validation dataset from the training
    #).cuda()

    
    # working training loop

    import os
    import time

    now =time.time()
    seconds = now


    from pathlib import Path

    my_file = Path(model_filename)
    if my_file.is_file():
        print('Using model file ' + model_filename)
        trainer.load(model_filename)

    import math
    import random
    import pickle

    no_of_chunks = math.ceil(len(input0)/chunk)
    input0_dask = da.from_array(input0,chunks=chunk)
    input2_dask = da.from_array(input2,chunks=chunk)

    #print(input0.shape)
    #print(input2.shape)
    
    images_total = []

    i = 0
    while(True):

        input0 = np.array(input0_dask.partitions[i:i+1])
        input2 = np.array(input2_dask.partitions[i:i+1])

        #print(input0.shape)
        #print(input2.shape)

        input0 = torch.from_numpy(input0)
        input0 = input0.to(torch.float)


        input2 = torch.from_numpy(input2)
        input2 = input2.to(torch.float)

        #print(input0.shape)
        #print(input2.shape)


        if(ignore_image_guide):
            start_image_or_video = None
        else:
            start_image_or_video = input2

    
        images = trainer.sample(text_embeds=input0,start_image_or_video = start_image_or_video,start_at_unet_number = unet_to_stop_sample - 1
                                ,stop_at_unet_number=unet_to_stop_sample,batch_size = 1, return_pil_images = True,cond_scale=cond_scale) # returns List[Image]


        images_total.extend(images)
        print("Resting...")
        time.sleep(60)

        #if not (i % save_model_every):
        #    trainer.save(model_filename)

        del input0
        del images
        del input2
        gc.collect()
    
        #print(no_of_chunks)
        #print(i)
        i = i+1
        if(i == no_of_chunks):
            break
    
    #print(images_total)
    try:
        os.remove('evaluate_imagen2.tmp')
    except:
        pass
    
    with open('evaluate_imagen2.tmp', 'wb') as handle:
        pickle.dump(images_total, handle)

    
