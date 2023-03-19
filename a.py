import torch
from imagen_pytorch import Unet, Imagen
from PIL import Image
import numpy as np
import torchvision
import torchvision.transforms as T
from torchvision.io import read_image

# unet for imagen
# TODO Document conda env ds2f_m_i, installation of cuda or rocm then torch them iamge_pytorch
# TODO forgo submodules structure

# mock images (get a lot of this) and text encodings from large T5

text_embeds = torch.randn(4, 256, 768).cuda()
print(text_embeds)

images = torch.randn(4, 3, 256, 256).cuda()
print(images)


unet1 = Unet(
    dim = 32,
    cond_dim = 512,
    dim_mults = (1, 2, 4, 8),
    num_resnet_blocks = 3,
    layer_attns = (False, True, True, True),
    layer_cross_attns = (False, True, True, True)
)

unet2 = Unet(
    dim = 32,
    cond_dim = 512,
    dim_mults = (1, 2, 4, 8),
    num_resnet_blocks = (2, 4, 8, 8),
    layer_attns = (False, False, False, True),
    layer_cross_attns = (False, False, False, True)
)

# imagen, which contains the unets above (base unet and super resoluting ones)

imagen = Imagen(
    unets = (unet1, unet2),
    image_sizes = (64, 256),
    timesteps = 1000,
    cond_drop_prob = 0.1
).cuda()



# feed images into imagen, training each unet in the cascade

for i in (1, 2):
    loss = imagen(images, text_embeds = text_embeds, unet_number = i)
    loss.backward()

# do the above for many many many many steps
# now you can sample an image based on the text embeddings from the cascading ddpm

images = imagen.sample(texts = [
    'a whale breaching from afar',
    'young girl blowing out candles on her birthday cake',
    'fireworks with blue and green sparkles'
], cond_scale = 3.)

#images.shape # (3, 3, 256, 256)

im_numpy = images.cpu().numpy()
im1 = np.transpose(im_numpy[0], (1,2,0))
im1f  = (im1 * 255 / np.max(im1)).astype('uint8')
x = T.ToPILImage()(im1f)
x.show()