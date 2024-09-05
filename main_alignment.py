import data_alignment.alignment as alignment
from data_reader.reader import reader

import sys

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import time


root_dir = Path(r'C:\Users\lenna\Documents\GitHub\optical-flow-analysis') #path to repository
input_dir = Path(r'data\LaserCutting\input\S3B4.avi') #"C:\Users\lenna\Documents\GitHub\optical-flow-analysis\data\LaserCutting\input\S1B1\registered_video.avi"
output_dir = Path(r'data\LaserCutting\S1A3\Aligned')
input_reader = reader(root_dir, input_dir)
image_stack = input_reader.read_avi()

#registered_stack = alignment.align_image_stack(image_stack)
registered_stack = alignment.align_image_stack_to_reference(image_stack, image_stack[500,...])


t, y, x = registered_stack.shape
out = cv2.VideoWriter('registered_video_ref500.avi', cv2.VideoWriter_fourcc(*'MJPG'), 20.0, (x, y))

for i in range(t):
    # Convert image to uint8 format
    frame = cv2.normalize(registered_stack[i], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    out.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))

out.release()