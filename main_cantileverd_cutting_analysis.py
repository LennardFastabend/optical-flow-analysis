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
input_dir = Path(r'data\CleftCut\input\S1 45 single tissue 1.avi') #"C:\Users\lenna\Documents\GitHub\optical-flow-analysis\data\CleftCut\input\S1 45 single tissue 1.avi"
output_dir = Path(r'data\CleftCut\S1 45 single tissue 1')
input_reader = reader(root_dir, input_dir)
image_stack = input_reader.read_avi()
t, y, x = image_stack.shape


#plt.imshow(image_stack[380:880,...], cmap='gray')

image_stack = image_stack[380:880, ...] # discard the first frames 

t, y, x = image_stack.shape

#image_stack = image_stack[:, 100:y-100, 60:x-40] #crop the image


dT=10
Tmax = 500-1

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

image_generator = visualizer(root_dir, output_dir/Path('images'))
#flowfield_generator = visualizer(root_dir, output_dir/Path('flowfields'))
defmap_generator = visualizer(root_dir, output_dir/Path('defmap'))
defmapRGB_generator = visualizer(root_dir, output_dir/Path('defmapRGB'))
#save the color map for the rgb_defmap seperately (after video generation)
max_magnitude = 0.4


for T in np.arange(0,Tmax-dT,5):
    print(T)

    image = image_stack[T,...]

    # Calculate a Deformation Map
    meanflowfield = opflow.MeanFlowField(flowfield_stack[T:T+dT,...])
    defmap = opflow.calculateMagnitude(meanflowfield)
    image_generator.saveImage(image_stack[T,...], title='Cleft at Time: '+str(T), filename='Image'+str(T))
    defmap_generator.saveDeformationMap(defmap, min=0, max=max_magnitude, title='DefMap at Time: '+str(T)+'-'+str(T+dT), filename='DefMap'+str(T))
    defmapRGB_generator.saveDeformationMapRGB(meanflowfield, max_magnitude, title='DefMap at Time: '+str(T)+'-'+str(T+dT), filename='DefMap'+str(T))
    #flowfield_generator.saveFlowField(image_stack[T,...], meanflowfield, title='FlowField at Time: '+str(T)+'-'+str(T+dT), filename='FlowField'+str(T), step=20, epsilon=0)


print("--- Runtime: %s seconds ---" % (time.time() - start_time))
print()

print('Generate Videos')
image_generator.create_video(fps=10)
#flowfield_generator.create_video(fps=10)
defmap_generator.create_video(fps=10)
defmapRGB_generator.create_video(fps=10)

defmapRGB_generator.saveHSVcolormap(max_magnitude, filename='colormap')