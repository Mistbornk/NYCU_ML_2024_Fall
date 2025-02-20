import numpy as np
from PIL import Image
import glob
import os

IMAGE_ID = 2
NUM_CLUSTER = 4
OUTPUT_DIR = f'./forGIF'

frames = []
imgs = sorted(glob.glob(os.path.join(OUTPUT_DIR, "*.png")))
for i in imgs:
    new_frame = Image.open(i)
    frames.append(new_frame)
    
# Save into a GIF file that loops forever
if frames:
    frames[0].save(os.path.join(OUTPUT_DIR, f'spectral_ratio_k++_img{IMAGE_ID}_{NUM_CLUSTER}_clusters.gif'),
                   format='GIF',
                   append_images=frames[1:],
                   save_all=True,
                   duration=250,
                   loop=0)