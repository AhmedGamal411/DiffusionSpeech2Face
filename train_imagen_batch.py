import numpy as np
from PIL import Image
import torch

def train_batch_unet1(input,output,model_filename,sub_epochs,batch_size,sample_every,save_model_every,image_size,unet_dim,unet1_image_size):
    from torch.utils.data import TensorDataset, DataLoader

    print("Training Unet No. 1")

    
    input = torch.from_numpy(input)
    input = input.to(torch.float)

    output = torch.from_numpy(output)
    output = output.to(torch.float)

    input = input.squeeze()
    output = output.squeeze()

    my_dataset = TensorDataset(output,input) # create your datset
    my_dataloader = DataLoader(my_dataset) # create your dataloader



    try:
        from imagen_pytorch import Unet, Imagen, ImagenTrainer
        from imagen_pytorch.data import Dataset
        from imagen_pytorch.t5 import t5_encode_text
        t5_encode_text(texts="Hi how have you been my love, what is the largest embedding that I can get here in egypt, is there any bigger or smaller than this i wonder").max()
    except:
        from imagen_pytorch import Unet, Imagen, ImagenTrainer
        from imagen_pytorch.data import Dataset
        from imagen_pytorch.t5 import t5_encode_text
        t5_encode_text(texts="Hi how have you been my love, what is the largest embedding that I can get here in egypt, is there any bigger or smaller than this i wonder").max()

    ground_truth = output[0].cpu().detach().numpy()
    ground_truth = np.moveaxis(ground_truth, 0, -1)
    ground_truth = ground_truth * 255
    ground_truth = ground_truth.astype(np.uint8)
    ground_truth = Image.fromarray(ground_truth)

    from imagen_pytorch import Unet, Imagen, ImagenTrainer
    from imagen_pytorch.data import Dataset

    # unets for unconditional imagen


    unet1 = Unet(
        dim = unet_dim,
        cond_dim = 512,
        dim_mults = (1, 2, 4, 8),
        num_resnet_blocks = 3,
        layer_attns = (False, True, True, True),
        layer_cross_attns = (False, True, True, True)
    )

    unet2 = Unet(
        dim = unet_dim,
        cond_dim = 512,
        dim_mults = (1, 2, 4, 8),
        num_resnet_blocks = (2, 4, 8, 8),
        layer_attns = (False, False, False, True),
        layer_cross_attns = (False, False, False, True)
    )

    #unet = Unet(
    #    dim = 32,
    #    dim_mults = (1, 2, 4, 8),
    #    num_resnet_blocks = 1,
    #    layer_attns = (False, False, False, True),
    #    layer_cross_attns = False
    #)

    # imagen, which contains the unet above

    #imagen = Imagen(
    #    unets = unet,
    #    image_sizes = 32,
    #    timesteps = 1000
    #)

    imagen = Imagen(
        unets = (unet1, unet2),
        image_sizes = (unet1_image_size, image_size),
        timesteps = 1000,
        cond_drop_prob = 0.1
    ).cuda()


    trainer = ImagenTrainer(
        imagen = imagen,
        split_valid_from_train = True # whether to split the validation dataset from the training
    ).cuda()

    # instantiate your dataloader, which returns the necessary inputs to the DDPM as tuple in the order of images, text embeddings, then text masks. in this case, only images is returned as it is unconditional training



    trainer.add_train_dataset(my_dataset, batch_size = batch_size * 8)

    # working training loop

    import os
    import time

    if not os.path.exists('imagen-samples'):
        os.makedirs('imagen-samples')

    now =time.time()
    seconds = now
    if not os.path.exists(str(seconds)):
        os.makedirs('imagen-samples/' + str(seconds))

    ground_truth.save('imagen-samples' + '/' + str(seconds) + '/ground_truth.png')

    from pathlib import Path

    my_file = Path(model_filename)
    if my_file.is_file():
        print('Using model file ' + model_filename)
        trainer.load(model_filename)

    import math

    for i in range(sub_epochs):
        loss = trainer.train_step(unet_number = 1, max_batch_size = batch_size)
        #print(f'loss: {loss}')

        if not (i % 50):
            valid_loss = trainer.valid_step(unet_number = 1, max_batch_size =  batch_size)
            print(f'valid loss: {valid_loss}')

        if not (i % sample_every) and trainer.is_main: # is_main makes sure this can run in distributed
            images = trainer.sample(text_embeds=input[:1, :],stop_at_unet_number=1,batch_size = 1, return_pil_images = True) # returns List[Image]
            images[0].save('imagen-samples' + '/' + str(seconds) + f'/sample-{i // 100}.png')

        if not (i % save_model_every):
            trainer.save(model_filename)

    trainer.save(model_filename)


def train_batch_unet2(input,output,model_filename,sub_epochs,batch_size,sample_every,save_model_every,image_size,unet_dim,unet1_image_size):
    from torch.utils.data import TensorDataset, DataLoader

    print("Training Unet No. 2")

    input = torch.from_numpy(input)
    input = input.to(torch.float)

    output = torch.from_numpy(output)
    output = output.to(torch.float)

    input = input.squeeze()
    output = output.squeeze()

    my_dataset = TensorDataset(output,input) # create your datset
    my_dataloader = DataLoader(my_dataset) # create your dataloader



    try:
        from imagen_pytorch import Unet, Imagen, ImagenTrainer
        from imagen_pytorch.data import Dataset
        from imagen_pytorch.t5 import t5_encode_text
        t5_encode_text(texts="Hi how have you been my love, what is the largest embedding that I can get here in egypt, is there any bigger or smaller than this i wonder").max()
    except:
        from imagen_pytorch import Unet, Imagen, ImagenTrainer
        from imagen_pytorch.data import Dataset
        from imagen_pytorch.t5 import t5_encode_text
        t5_encode_text(texts="Hi how have you been my love, what is the largest embedding that I can get here in egypt, is there any bigger or smaller than this i wonder").max()

    ground_truth = output[0].cpu().detach().numpy()
    ground_truth = np.moveaxis(ground_truth, 0, -1)
    ground_truth = ground_truth * 255
    ground_truth = ground_truth.astype(np.uint8)
    ground_truth = Image.fromarray(ground_truth)

    from imagen_pytorch import Unet, Imagen, ImagenTrainer
    from imagen_pytorch.data import Dataset

    # unets for unconditional imagen


    unet1 = Unet(
        dim = unet_dim,
        cond_dim = 512,
        dim_mults = (1, 2, 4, 8),
        num_resnet_blocks = 3,
        layer_attns = (False, True, True, True),
        layer_cross_attns = (False, True, True, True)
    )

    unet2 = Unet(
        dim = unet_dim,
        cond_dim = 512,
        dim_mults = (1, 2, 4, 8),
        num_resnet_blocks = (2, 4, 8, 8),
        layer_attns = (False, False, False, True),
        layer_cross_attns = (False, False, False, True)
    )

    #unet = Unet(
    #    dim = 32,
    #    dim_mults = (1, 2, 4, 8),
    #    num_resnet_blocks = 1,
    #    layer_attns = (False, False, False, True),
    #    layer_cross_attns = False
    #)

    # imagen, which contains the unet above

    #imagen = Imagen(
    #    unets = unet,
    #    image_sizes = 32,
    #    timesteps = 1000
    #)

    imagen = Imagen(
        unets = (unet1, unet2),
        image_sizes = (unet1_image_size, image_size),
        timesteps = 1000,
        cond_drop_prob = 0.1
    ).cuda()


    trainer = ImagenTrainer(
        imagen = imagen,
        split_valid_from_train = True # whether to split the validation dataset from the training
    ).cuda()

    # instantiate your dataloader, which returns the necessary inputs to the DDPM as tuple in the order of images, text embeddings, then text masks. in this case, only images is returned as it is unconditional training



    trainer.add_train_dataset(my_dataset, batch_size = batch_size * 8)

    # working training loop

    import os
    import time

    if not os.path.exists('imagen-samples'):
        os.makedirs('imagen-samples')

    now =time.time()
    seconds = now
    if not os.path.exists(str(seconds)):
        os.makedirs('imagen-samples/' + str(seconds))

    ground_truth.save('imagen-samples' + '/' + str(seconds) + '/ground_truth.png')

    from pathlib import Path

    my_file = Path(model_filename)
    if my_file.is_file():
        print('Using model file ' + model_filename)
        trainer.load(model_filename)

    import math

    for i in range(sub_epochs): 
        loss = trainer.train_step(unet_number = 2, max_batch_size = batch_size)
        #print(f'loss: {loss}')

        if not (i % 50):
            valid_loss = trainer.valid_step(unet_number = 2, max_batch_size = batch_size)
            print(f'valid loss: {valid_loss}')

        if not (i % sample_every) and trainer.is_main: # is_main makes sure this can run in distributed
            images = trainer.sample(text_embeds=input[:1, :],batch_size = 1, return_pil_images = True) # returns List[Image]
            images[0].save('imagen-samples' + '/' + str(seconds) + f'/sample-{i // 100}.png')

        if not (i % save_model_every):
            trainer.save(model_filename)

    trainer.save(model_filename)