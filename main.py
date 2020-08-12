import argparse
import glob
import os
import sys
sys.path.append('./HiDT')

import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from hidt.networks.enhancement.RRDBNet_arch import RRDBNet
from hidt.style_transformer import StyleTransformer
from hidt.utils.preprocessing import GridCrop, enhancement_preprocessing
from hidt.utils.io import save_img

config_path = './configs/daytime.yaml'
gen_weights_path = './trained_models/generator/daytime.pt'
inference_size = 256  # the network has been trained to do inference in 256px, any higher value might lead to artifacts
device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
image_path = './images/daytime/content/0.jpg'
styles_path = './styles.txt'
enhancer_weights = './trained_models/enhancer/enhancer.pth'
result_path = './results'
style = 'hard_day'

style_transformer = StyleTransformer(config_path,
                                     gen_weights_path,
                                     inference_size=inference_size,
                                     device=device)
with open(styles_path) as f:
    styles = f.read()
styles = {style.split(',')[0]: torch.tensor([float(el) for el in style.split(',')[1][1:-1].split(' ')]) for style in styles.split('\n')[:-1]}
image = Image.open(image_path)
crop_transform = GridCrop(4, 1, hires_size=inference_size * 4)

style_to_transfer = styles[style]

style_to_transfer = style_to_transfer.view(1, 1, 3, 1).to(device)
with torch.no_grad():
    content_decomposition = style_transformer.get_content(image)[0]
    decoder_input = {'content': content_decomposition['content'],
                     'intermediate_outputs': content_decomposition['intermediate_outputs'],
                     'style': style_to_transfer}
    transferred = style_transformer.trainer.gen.decode(decoder_input)['images']

result_images = transforms.ToPILImage()((transferred[0].cpu().clamp(-1, 1) + 1.) / 2.)

os.makedirs(result_path, exist_ok=True)

source_name = image_path.split('/')[-1].split('.')[0]
style_name = style

save_img(result_images,
                         os.path.join(result_path,
                                      source_name + '_to_' + style_name + '.jpg')
                         )