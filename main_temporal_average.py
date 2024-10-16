from data_reader.reader import reader
from output_generation.visualizer import visualizer
import data_analysis.optical_flow as opflow
import data_analysis.geometric_quantification as geoquant
from data_segmentation.segmentation import Segmentation

import sys

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd

root_dir = Path(r'C:\Users\lenna\Documents\GitHub\optical-flow-analysis') #path to repository

input_dir = Path(r'data\PhaseContrastCleft\P01\input\Aligned\LinearStackAlignmentSift_Gauss5px.avi')
#input_dir = Path(r'data\PhaseContrastCleft\P02\input\P08#39_live_W03-P02_aligned.avi')
#input_dir = Path(r'data\PhaseContrastCleft\P06\input\P08#39_live_W03-P06.avi') #Read in Aligned Data!

#input_dir = Path(r'data\PhaseContrastCleft\ContractilityComparisson\Blebbistatin\input\P08#39_live_W08-P01.avi')

output_dir = Path(r'data\PhaseContrastCleft\P01\temporal_average') 



input_reader = reader(root_dir, input_dir)
image_stack = input_reader.read_avi()

### Crop the image (resolves issues due to alignment of the images)
t, y, x = image_stack.shape
image_stack = image_stack[:, 100:y-100, 100:x-10] #crop the image and select the time



### Optional temporal averaging of the image_stack:
averaged_images = []

'''
###Simple Weighted Smoothing
for t in range(1,image_stack.shape[0]-1):
    avg_img = (image_stack[t-1,...]/2 + image_stack[t,...] + image_stack[t+1,...]/2) / 2 # 1/4 + 1/2 + 1/4
    averaged_images.append(avg_img)
image_stack_avg = np.stack(averaged_images, axis=0)
print(image_stack_avg.shape)
'''

###Heavy Weighted Smoothing
for t in range(2,image_stack.shape[0]-2):
    avg_img = (image_stack[t-2,...]/10 + image_stack[t-1,...]/5 + image_stack[t,...]*(2/5) + image_stack[t+1,...]/5 + image_stack[t+2,...]/10) # 1/10 + 1/5 + 2/5 + 1/5 + 1/10
    averaged_images.append(avg_img)
image_stack_avg = np.stack(averaged_images, axis=0)
print(image_stack_avg.shape)


### calculate Example FlowFields for the defined time
Tmax = 100
farneback_parameters = {"pyr_scale": 0.5,
                        "levels": 3,
                        "winsize": 5,
                        "iterations": 3,
                        "poly_n": 5, #10,
                        "poly_sigma": 1.2,
                        "flags": 0}

print('Start Farneback Analysis')
dt_OptFlow = 1
flowfield_stack = opflow.FlowFieldStack(image_stack_avg, farneback_parameters, t0=0, tfin=Tmax, dt=dt_OptFlow)
print('Farneback Analysis Finished')
print(flowfield_stack.shape)
print()


T0=0
step = 1
temp_scale = Tmax

max_magnitude = 10

image_generator = visualizer(root_dir, output_dir/Path('images'))
flowfield_generator = visualizer(root_dir, output_dir/Path('flowfields'))
#defmap_generator = visualizer(root_dir, output_dir/Path('defmap'))
defmap_rgb_generator = visualizer(root_dir, output_dir/Path('defmap_rgb'))

for T in np.arange(T0,T0+temp_scale,step):
    print(T)

    image = image_stack_avg[T,...]
    flowfield = flowfield_stack[T,...]
    defmap = opflow.calculateMagnitude(flowfield)

    #defmap_generator.saveDeformationMap(defmap, min=0, max=max_magnitude, title='DefMap at Time: '+str(T)+'-'+str(T+dt_OptFlow), filename='DefMap'+str(T))
    defmap_rgb_generator.saveDeformationMapRGB(flowfield, max_magnitude, title='DefMap at Time: '+str(T)+'-'+str(T+dt_OptFlow), filename='DefMap'+str(T))
    image_generator.saveImage(image, title='Averaged Image: '+str(T), filename='AvgImg'+str(T))
    flowfield_generator.saveFlowField(image, flowfield, title='FlowField at Time: '+str(T)+'-'+str(T+dt_OptFlow), filename='FlowField'+str(T), step=10, epsilon=0, scale=0.5)

#defmap_generator.create_video()
defmap_rgb_generator.create_video(fps=10)
flowfield_generator.create_video(fps=10)
image_generator.create_video(fps=10)
