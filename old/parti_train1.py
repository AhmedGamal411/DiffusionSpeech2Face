from parti_pytorch import VitVQGanVAE, VQGanVAETrainer


import os
import configparser

# Loading configurations
configParser = configparser.RawConfigParser()   
configFilePath = r'configuration.txt'
configParser.read(configFilePath)
datasetPathVideo =  configParser.get('COMMON', 'datasetPathVideo')

vit_vae = VitVQGanVAE(
    dim = 256,               # dimensions
    image_size = 256,        # target image size
    patch_size = 16,         # size of the patches in the image attending to each other
    num_layers = 3           # number of layers
).cuda()


trainer = VQGanVAETrainer(
    vit_vae,
    folder = '/media/gamal/Passport/Datasets/VoxCeleb2Test/Voxceleb2TestFaces',
    num_train_steps = 100,
    lr = 3e-4,
    batch_size = 4,
    grad_accum_every = 8,
    amp = True,
    results_folder = '/media/gamal/Passport/parti/vae',
    save_results_every = 10,
    save_model_every = 50,
)

trainer.train()

vit_vae.enco