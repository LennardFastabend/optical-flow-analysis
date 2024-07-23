from data_reader.reader import reader
from output_generation.visualizer import visualizer
import data_analysis.optical_flow as opflow
#import data_analysis.geometric_quantification as geoquant
#from data_segmentation.segmentation import Segmentation

import sys

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import time

start_time = time.time()



root_dir = Path(r'C:\Users\lenna\Documents\GitHub\optical-flow-analysis') #path to repository
input_dir = Path(r'data\PrimaryCantileveredCleft\input\P08#39_live_W01-P01_aligned.avi') 
output_dir = Path(r'data\PrimaryCantileveredCleft\P08#39_live_W01-P01')
input_reader = reader(root_dir, input_dir)
image_stack = input_reader.read_avi()

t, y, x = image_stack.shape

image_stack = image_stack[:, 100:y-100, 50:x-50] #crop the image

dT=7
Tmax = 200

farneback_parameters = {"pyr_scale": 0.5,
                        "levels": 3,
                        "winsize": 5,#15,
                        "iterations": 3,
                        "poly_n": 5,
                        "poly_sigma": 1.2,
                        "flags": 0}

print('Start Farneback Analysis')
flowfield_stack = opflow.FlowFieldStack(image_stack, farneback_parameters, t0=0, tfin=Tmax, dt=1)
print('Farneback Analysis Finished')

flowfield_generator = visualizer(root_dir, output_dir/Path('flowfields'))
defmap_generator = visualizer(root_dir, output_dir/Path('defmap'))
defmapRGB_generator = visualizer(root_dir, output_dir/Path('defmapRGB'))
#save the color map for the rgb_defmap seperately
max_magnitude = 10
defmapRGB_generator.saveHSVcolormap(max_magnitude, filename='colormap')


for T in np.arange(0,Tmax-dT,1):

    image = image_stack[T,...]

    # Calculate a Deformation Map
    meanflowfield = opflow.MeanFlowField(flowfield_stack[T:T+dT,...])
    defmap = opflow.calculateMagnitude(meanflowfield)
    defmap_generator.saveDeformationMap(defmap, min=0, max=10, title='DefMap at Time: '+str(T)+'-'+str(T+dT), filename='DefMap'+str(T))
    defmapRGB_generator.saveDeformationMapRGB(meanflowfield, max_magnitude, title='DefMap at Time: '+str(T)+'-'+str(T+dT), filename='DefMap'+str(T))
    flowfield_generator.saveFlowField(image_stack[T,...], meanflowfield, title='FlowField at Time: '+str(T)+'-'+str(T+dT), filename='FlowField'+str(T), step=20, epsilon=0)


print("--- Runtime: %s seconds ---" % (time.time() - start_time))
print()