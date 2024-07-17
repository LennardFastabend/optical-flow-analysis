from data_reader.reader import reader
from output_generation.visualizer import visualizer
import data_analysis.optical_flow as opflow
import data_analysis.geometric_quantification as geoquant
from data_segmentation.segmentation import SegementCleft
from data_segmentation.segmentation import SegementGrowthFront

import sys

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd

root_dir = Path(r'C:\Users\lenna\Documents\GitHub\optical-flow-analysis') #path to repository
input_dir = Path(r'data\PhaseContrastCleft\P01\input\Aligned\LinearStackAlignmentSift_Gauss5px.avi') #Read in Aligned Data!
output_dir = Path(r'data\PhaseContrastCleft\P01\example_images')

input_reader = reader(root_dir, input_dir)
image_stack = input_reader.read_avi()

### Crop the image (resolves issues due to alignment of the images)
t, y, x = image_stack.shape
crop_size = 25
image_stack = image_stack[:, crop_size:y-crop_size, crop_size:x-crop_size]

### calculate Example FlowFields for the defined time
dT=1
Tmax = 35
max_magnitude = 10

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
print()

'''
image_generator = visualizer(root_dir, output_dir/Path('images'))
flowfield_generator = visualizer(root_dir, output_dir/Path('flowfields'))
defmap_generator = visualizer(root_dir, output_dir/Path('defmap'))
defmap_rgb_generator = visualizer(root_dir, output_dir/Path('defmap_rgb'))
#segmentation_generator = visualizer(root_dir, output_dir/Path('segmentation'))
#geoquant_generator = visualizer(root_dir, output_dir/Path('geometric_quantification'))


#save the color map for the rgb_defmap seperately
defmap_rgb_generator.saveHSVcolormap(max_magnitude, filename='colormap')
### T-Scan
for T in np.arange(0,Tmax):
    print('Output Time:', T)
    image = image_stack[T,...]
    #front_mask,filtered_image_front,front_contour_line = SegementGrowthFront(image)
    #cleft_mask,filtered_image_cleft,cleft_contour_line = SegementCleft(image)
    #segmentation_generator.saveSegmentationMasks(image, front_contour_line, cleft_contour_line, title='Segmentation at Time:'+str(T), filename='segmentation'+str(T))

    meanflowfield = opflow.MeanFlowField(flowfield_stack[T:T+dT,...])
    defmap = opflow.calculateMagnitude(meanflowfield)

    image_generator.saveImage(image_stack[T,...], title='Cleft at Time: '+str(T), filename='Image'+str(T))
    defmap_generator.saveDeformationMap(defmap, min=0, max=max_magnitude, title='DefMap at Time: '+str(T)+'-'+str(T+dT), filename='DefMap'+str(T))
    defmap_rgb_generator.saveDeformationMapRGB(meanflowfield, max_magnitude, title='DefMap at Time: '+str(T)+'-'+str(T+dT), filename='DefMap'+str(T))
    flowfield_generator.saveFlowField(image_stack[T,...], meanflowfield, title='FlowField at Time: '+str(T)+'-'+str(T+dT), filename='FlowField'+str(T), step=15, epsilon=0)
'''
########################################
### Example Spatio-Temporal Blurring ###
########################################
'''
defmap_generator = visualizer(root_dir, output_dir/Path('spatiotemporal_scales'))

T = 25
dT_array = [1,3,5,10,15]
kernel_sizes = [0,5,15,25,45]

for dT in dT_array:
    for kernel_size in kernel_sizes:
        meanflowfield = opflow.MeanFlowField(flowfield_stack[T:T+dT,...])
        smoothedmeanflowfield = opflow.BlurFlowField(meanflowfield,kernel_size)

        defmap = opflow.calculateMagnitude(smoothedmeanflowfield)

        defmap_generator.saveDeformationMap(defmap, min=0, max=max_magnitude, title='DefMap: dT='+str(dT)+'; Kernelsize='+ str(kernel_size), filename='DefMap_dT'+str(dT)+'_ksize'+str(kernel_size))
'''

div_generator = visualizer(root_dir, output_dir/Path('divergence'))

T = 25
dT = 1

meanflowfield = opflow.MeanFlowField(flowfield_stack[T:T+dT,...])
div = opflow.Divergence(meanflowfield)

div_generator.saveDivergence(div, title='Divergence at Time: '+str(T)+'-'+str(T+dT), filename='Divergence'+str(T))

